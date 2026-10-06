package com.convaiinnovations.laya.hooks;

/**
 * The one place a caller's own code runs inside a prediction.
 *
 * <p>Override only the events you need; the rest are no-ops. A hook may watch a call, rewrite
 * what it is about to ask, or answer it outright with {@link PredictContext#skip}.
 *
 * <pre>{@code
 * final class Tracer implements Hook {
 *     @Override public void onPredictEnd(PredictContext ctx) {
 *         span.record(ctx.usage().inputTokens(), ctx.elapsedMs());
 *     }
 * }
 * agent.hooks().addHook(new Tracer());
 * }</pre>
 *
 * <p>This is ONE type where the reference has two. {@code laya.hooks.Hook} is a structural
 * {@code Protocol} — a hook is anything with at least one of the six methods — and
 * {@code BaseHook} is the concrete no-op class to subclass when you would rather not implement
 * by shape. An interface with default methods is both at once, so there is nothing for a second
 * type to be. The consequence is that the reference's "missing methods are skipped" becomes "the
 * default does nothing", which is the same behaviour from the outside: {@link Hooks#dispatch}
 * calling a default no-op and the reference finding no attribute to call both leave the context
 * untouched and nothing in the log.
 *
 * <p>Nothing here mirrors the reference's {@code AsyncHook}, deliberately. That class exists to
 * solve a problem the JVM does not have: in Python an {@code async def} hook returns a coroutine
 * that only an event loop can finish, so a synchronous caller needs
 * {@code run_coroutine_sync} — and a background loop thread when it is already inside one — to
 * get a value back. A JVM method call is already synchronous, so a hook that wants to do
 * asynchronous work composes it and blocks on the result itself:
 *
 * <pre>{@code
 * @Override public void onPredictEnd(PredictContext ctx) {
 *     ship(ctx.results()).toCompletableFuture().join();   // or orTimeout(...), or ignore it
 * }
 * }</pre>
 *
 * <p>Porting {@code AsyncHook} would mean choosing a future type for every caller and owning a
 * thread pool to await it on, to wrap a {@code join} the caller can write in one line and bound
 * however their own runtime wants. Half-porting it — accepting a {@code CompletionStage} and
 * blocking on it with no deadline — would be worse: {@link Hooks.Policy#timeout()} already
 * bounds a slow hook, and a second, invisible wait inside one would make a hook that never
 * returns look like a hook that never started.
 *
 * <p>A hook runs on the calling thread, so it is as thread-safe as the call around it. Install
 * one that is not, and set {@link HookRegistry#concurrent(boolean)} to false to have dispatch
 * serialise it.
 */
public interface Hook {

    /**
     * Before inference, with the states and questions the call was made with.
     *
     * <p>Rewriting {@link PredictContext#states} or {@link PredictContext#questions} changes what
     * is asked; {@link PredictContext#skip} answers the call without the model running at all.
     */
    default void onPredictStart(PredictContext ctx) {
    }

    /**
     * After inference, on the success path and the failure path alike.
     *
     * <p>{@link PredictContext#results} is the answer and may be replaced; on a failure it is
     * null and {@link PredictContext#error} says why.
     */
    default void onPredictEnd(PredictContext ctx) {
    }

    /** After a router has chosen a checkpoint, before it is asked anything. */
    default void onRoute(PredictContext ctx) {
    }

    /** After a checkpoint has been loaded. */
    default void onLoad(PredictContext ctx) {
    }

    /** After a checkpoint has been evicted. */
    default void onEvict(PredictContext ctx) {
    }

    /**
     * When the call failed, before {@link #onPredictEnd}.
     *
     * <p>Throwing from here does not replace the failure that triggered it: see
     * {@link Hooks#around}.
     */
    default void onError(PredictContext ctx) {
    }
}
