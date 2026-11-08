import { Veya, triageQuestions } from 'veya-client';

const veya = new Veya({
  baseURL: process.env.VEYA_BASE_URL ?? 'http://127.0.0.1:8000',
  apiKey: process.env.VEYA_API_KEY,
  timeoutMs: 600_000,
});

try {
  const result = await veya.predict(
    { message: 'I was billed twice for March. Please refund the duplicate today.' },
    triageQuestions(),
  );
  console.log(JSON.stringify(result, null, 2));
} catch (error) {
  console.error(error);
  process.exitCode = 1;
}
