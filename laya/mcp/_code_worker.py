"""Disposable QuickJS capability sandbox. The pipe carries bounded JSON bytes, never objects.

No Python callable, module loader, operating-system library or ambient authority is
installed in the JavaScript context. Host operations are suspended JSON requests.
"""
import json
import os
import time


BOOTSTRAP = r"""
((source, input, names) => {
  const P = Promise, then = Function.call.bind(P.prototype.then), resolve = P.resolve.bind(P);
  const stringify = JSON.stringify.bind(JSON), parse = JSON.parse.bind(JSON);
  const finite = Number.isFinite;
  const mapSet = Function.call.bind(Map.prototype.set), mapGet = Function.call.bind(Map.prototype.get);
  const mapDelete = Function.call.bind(Map.prototype.delete);
  const mapSize = Function.call.bind(Object.getOwnPropertyDescriptor(Map.prototype, 'size').get);
  const push = Function.call.bind(Array.prototype.push), shift = Function.call.bind(Array.prototype.shift);
  const create = Object.create, freeze = Object.freeze;
  // Private Promise chaining must not consult guest-overridden constructors,
  // species, resolve or then hooks after the supplied function is parsed.
  freeze(P.prototype); freeze(P);
  const pending = new Map(), queue = [];
  let sequence = 0, completed = false, output, failure, encoding = false;
  const encode = value => stringify(value, (key, item) => {
    if (typeof item === 'undefined' || typeof item === 'function' ||
        typeof item === 'symbol' || typeof item === 'bigint' ||
        (typeof item === 'number' && !finite(item)))
      throw new TypeError('Only finite JSON values may cross the Veya boundary');
    return item;
  });
  const api = create(null);
  // These callbacks are unnecessary for decision composition and can surface raw
  // engine-job exceptions outside the normal Promise rejection path.
  for (const name of ['queueMicrotask', 'FinalizationRegistry', 'WeakRef'])
    Object.defineProperty(globalThis, name, {value: undefined, writable: false, configurable: false});
  for (const name of names) api[name] = args => {
    // Snapshot at the call boundary; later script mutation cannot rewrite a request.
    if (encoding) throw new TypeError('Veya calls cannot reenter argument serialization');
    let request, id = sequence + 1;
    encoding = true;
    try { request = encode({id, method: name, args: args === undefined ? {} : args}); }
    finally { encoding = false; }
    sequence = id;
    return new P((resolve, reject) => {
      mapSet(pending, id, {resolve, reject});
      push(queue, request);
    });
  };
  freeze(api);
  Object.defineProperty(globalThis, 'veya', {value: api, writable: false, configurable: false});
  // Indirect eval gives the function only the global realm, not this closure's state.
  let fn;
  try {
    fn = (0, eval)('(' + source + ')');
    if (typeof fn !== 'function') throw 0;
  } catch (_) { return () => '{"type":"error","code":"script_error"}'; }
  then(then(resolve(), () => fn(input, api)), value => {
    try { output = encode(value); completed = true; }
    catch (_) { failure = 'The script must return a finite JSON value'; completed = true; }
  }, () => { failure = 'JavaScript execution failed'; completed = true; });
  return response => {
    try {
    if (response !== null) {
      const message = parse(response), item = mapGet(pending, message.id);
      if (!item) throw new Error('Unknown response');
      mapDelete(pending, message.id);
      if (message.error) {
        const error = new Error(message.error.message);
        error.code = message.error.code;
        item.reject(error);
      } else item.resolve(message.result);
      return null;
    }
    if (completed) {
      if (mapSize(pending) || queue.length) return '{"type":"error","code":"unawaited_calls"}';
      return failure ? '{"type":"error","code":"script_error"}'
                     : '{"type":"result","value":' + stringify(output) + '}';
    }
    if (queue.length) return '{"type":"call","value":' + stringify(shift(queue)) + '}';
    return '{"type":"pending"}';
    } catch (_) { return '{"type":"error","code":"script_error"}'; }
  };
})
"""


def run_worker(connection, code, encoded_input, methods, limits):
    """Run one guest; parent owns wall-time, admission, validation and inference."""
    try:
        import quickjs

        context = quickjs.Context()
        context.set_memory_limit(limits["memory_bytes"])
        context.set_max_stack_size(limits["stack_bytes"])
        cpu_start = time.process_time()
        connection.send_bytes(json.dumps({"type": "ready", "cpu_started": cpu_start, "pid": os.getpid()}).encode("utf-8"))
        jobs = 0

        def bounded(call):
            remaining = limits["cpu_seconds"] - (time.process_time() - cpu_start)
            if remaining <= 0:
                raise RuntimeError("cpu_limit")
            context.set_time_limit(remaining)
            return call()

        # Object.__call__ installs the same interrupt as Context.eval. Never use
        # Context.get or Object.json on script objects: their getters lack interrupts.
        program = (BOOTSTRAP + "(" + json.dumps(code) + ",JSON.parse(" + json.dumps(encoded_input) + "),"
                   + json.dumps(methods) + ")")
        pump = bounded(lambda: context.eval(program))
        reply = None
        while True:
            if reply is not None:
                bounded(lambda: pump(reply))
                reply = None
            # Reach a microtask checkpoint before authorizing any queued host call.
            # This rejects fire-and-forget work after the main program has returned.
            while bounded(context.execute_pending_job):
                jobs += 1
                if jobs > limits["max_pending_jobs"]:
                    raise RuntimeError("job_limit")
            event = json.loads(bounded(lambda: pump(None)))
            if event["type"] == "pending":
                raise RuntimeError("unresolved_promise")
            raw = json.dumps(event, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            if len(raw) > limits["max_message_bytes"]:
                raise RuntimeError("message_limit")
            connection.send_bytes(raw)
            if event["type"] != "call":
                return
            reply = connection.recv_bytes(limits["max_message_bytes"]).decode("utf-8")
    except BaseException:
        # Engine errors may run guest toString/getters while being converted by
        # the binding. Parent wall-time still terminates that disposable process.
        try:
            connection.send_bytes(b'{"type":"error","code":"sandbox_error"}')
        except (OSError, EOFError):
            pass
    finally:
        connection.close()
