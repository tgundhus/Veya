// The HTTP client for a self-hosted `veya-serve`: three routes, no inference. Peer of the
// `veya-client` npm package. `java.net.http` is in the JDK, so this has no runtime dependency
// either; it shares the runtime's answer types.
dependencies {
    api(project(":veya-java"))
}
