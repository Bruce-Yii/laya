package com.convaiinnovations.laya.hooks;

import com.convaiinnovations.laya.Prediction;
import com.convaiinnovations.laya.Question;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.locks.Lock;
import java.util.function.Consumer;

/**
 * Composing, dispatching and bounding hooks — the parts of {@code laya.hooks} that are not a type.
 *
 * <p>Three things here are contracts rather than implementation details, and each is pinned by
 * {@code fixtures/hooks.json}:
 *
 * <ul>
 *   <li><b>Order.</b> {@link #compose} is process-wide defaults, then the hooks installed on the
 *       agent, then the per-call ones; {@link #dispatch} calls them in that order. A port that
 *       composes the other way round still runs every hook and fails nothing — except that the
 *       tracer a caller installed to watch what their per-call hook did now runs first and sees
 *       nothing.</li>
 *   <li><b>{@link PredictContext#skip} does not stop the chain.</b> It assigns the results; it is
 *       {@link #around} reading them afterwards that skips inference. Every later start hook
 *       still runs, and can overwrite the answer.</li>
 *   <li><b>{@link Policy#raiseErrors}.</b> True fails the request at the first throwing hook and
 *       the rest of the chain never runs. False reports and continues, which is what a telemetry
 *       hook needs — it must not be able to fail a request.</li>
 * </ul>
 */
public final class Hooks {

    private static final System.Logger LOG =
            System.getLogger(Hooks.class.getPackage().getName());

    private static final Object DEFAULTS_MUTEX = new Object();
    private static List<Hook> defaults = List.of();
    private static final ThreadLocal<Boolean> SKIP_DEFAULTS = ThreadLocal.withInitial(() -> false);

    private Hooks() {
    }

    /**
     * The six lifecycle events, in the reference's own order.
     *
     * <p>The order is data, not presentation: {@code HOOK_EVENTS} is what the reference's
     * refusal messages enumerate, and a port that reorders it describes a different API.
     */
    public enum Event {
        /** Before inference. */
        PREDICT_START("on_predict_start"),
        /** After inference, success or failure. */
        PREDICT_END("on_predict_end"),
        /** After a router chose a checkpoint. */
        ROUTE("on_route"),
        /** After a checkpoint was loaded. */
        LOAD("on_load"),
        /** After a checkpoint was evicted. */
        EVICT("on_evict"),
        /** When the call failed, before {@link #PREDICT_END}. */
        ERROR("on_error");

        private final String wireName;

        Event(String wireName) {
            this.wireName = wireName;
        }

        /** The name the reference calls this event, which is what a cross-language caller reads. */
        public String wireName() {
            return wireName;
        }

        void callOn(Hook hook, PredictContext ctx) {
            switch (this) {
                case PREDICT_START -> hook.onPredictStart(ctx);
                case PREDICT_END -> hook.onPredictEnd(ctx);
                case ROUTE -> hook.onRoute(ctx);
                case LOAD -> hook.onLoad(ctx);
                case EVICT -> hook.onEvict(ctx);
                case ERROR -> hook.onError(ctx);
            }
        }
    }

    /**
     * One call's totalled token usage, as a hook sees it.
     *
     * <p>Not {@link com.convaiinnovations.laya.Usage}: that describes ONE state, including what
     * its budget dropped and which question was blamed. This is the sum over the call, which is
     * the only number a hook can report without re-deriving per-state attribution the reference
     * does not give it either.
     */
    public record Totals(int inputTokens, int outputTokens) {
    }

    /**
     * What dispatch does with a hook that throws, one that overruns, and one that is not
     * re-entrant.
     *
     * @param raiseErrors true fails the call at the first throwing hook; false reports through
     *     {@code onFailure} and runs the rest of the chain
     * @param lock        held across each hook call, for hooks that are not safe to run
     *     concurrently, or null to run them unguarded
     * @param timeout     bounds EACH hook call, or null for no limit. A hook that overruns
     *     raises — see {@link #dispatch} for what that does and does not protect
     * @param onFailure   where a swallowed failure is reported when {@code raiseErrors} is
     *     false. Never null; {@link #reporting()} is the default and logs at WARNING
     */
    public record Policy(boolean raiseErrors, Lock lock, Duration timeout,
                         Consumer<String> onFailure) {

        public Policy {
            timeout = validateTimeout(timeout);
            if (onFailure == null) {
                throw new IllegalArgumentException(
                        "a policy needs somewhere to report a swallowed hook failure; pass "
                        + "Hooks.reporting() for the default");
            }
        }

        /** Fail the call on a throwing hook, no lock, no timeout. The reference's defaults. */
        public static Policy raising() {
            return new Policy(true, null, null, reporting());
        }

        /** This policy with {@code raiseErrors} replaced. */
        public Policy raiseErrors(boolean value) {
            return new Policy(value, lock, timeout, onFailure);
        }

        /** This policy with the serialising lock replaced. */
        public Policy lock(Lock value) {
            return new Policy(raiseErrors, value, timeout, onFailure);
        }

        /** This policy with the per-hook timeout replaced. */
        public Policy timeout(Duration value) {
            return new Policy(raiseErrors, lock, value, onFailure);
        }

        /** This policy reporting swallowed failures somewhere else — a test, or a metric. */
        public Policy onFailure(Consumer<String> value) {
            return new Policy(raiseErrors, lock, timeout, value);
        }
    }

    /** The default failure sink: one WARNING line per swallowed hook failure. */
    public static Consumer<String> reporting() {
        return message -> LOG.log(System.Logger.Level.WARNING, message);
    }

    /** What inference {@link #around} runs when no hook has answered the call. */
    @FunctionalInterface
    public interface Inference {

        /**
         * Answers the call.
         *
         * @param states     the states the start hooks left on the context, never empty
         * @param questions  the questions they left, never empty of meaning
         * @param maxLen     a per-call token budget, or null for the checkpoint's own
         * @param headMaxLen a per-call head budget, or null for the checkpoint's own
         */
        List<Prediction> run(List<Object> states, Map<String, Question> questions,
                             Integer maxLen, Integer headMaxLen);
    }

    /**
     * Wraps a hook object around a plain {@code onPredictStart} callback.
     *
     * <p>{@link Hook} has six methods, so it is not a functional interface and a lambda cannot
     * be one. This is the reference's {@code on_predict_start=} parameter: the common case is a
     * caller who wants one event, and making them write an anonymous class for it is the kind of
     * friction that stops a tracer being added at all.
     */
    public static Hook onPredictStart(Consumer<PredictContext> callback) {
        requireCallback(callback, "onPredictStart");
        return new Hook() {
            @Override
            public void onPredictStart(PredictContext ctx) {
                callback.accept(ctx);
            }

            @Override
            public String toString() {
                return "onPredictStart(" + callback + ")";
            }
        };
    }

    /** Wraps a hook object around a plain {@code onPredictEnd} callback. */
    public static Hook onPredictEnd(Consumer<PredictContext> callback) {
        requireCallback(callback, "onPredictEnd");
        return new Hook() {
            @Override
            public void onPredictEnd(PredictContext ctx) {
                callback.accept(ctx);
            }

            @Override
            public String toString() {
                return "onPredictEnd(" + callback + ")";
            }
        };
    }

    private static void requireCallback(Consumer<PredictContext> callback, String name) {
        if (callback == null) {
            throw new IllegalArgumentException(name + " must not be null");
        }
    }

    /**
     * One ordered list from a hook list and the two convenience callbacks.
     *
     * <p>Hooks first, in their own order, then the start callbacks, then the end callbacks —
     * so a {@code onPredictEnd} callback never runs before a hook object's own end method.
     *
     * <p>Three of the reference's refusals have no counterpart here and are not omissions: a
     * class rather than an instance, an object implementing none of the six events, and an
     * event attribute that is not callable are all unrepresentable once {@link Hook} is a type.
     * The fixture records them anyway, so that a reference that stopped refusing them would be
     * caught rather than quietly leaving this paragraph wrong. A null entry IS representable,
     * and is refused here.
     */
    public static List<Hook> normalise(List<? extends Hook> hooks,
                                       List<? extends Consumer<PredictContext>> onPredictStart,
                                       List<? extends Consumer<PredictContext>> onPredictEnd) {
        List<Hook> out = new ArrayList<>();
        if (hooks != null) {
            for (Hook hook : hooks) {
                if (hook == null) {
                    throw new IllegalArgumentException("a hooks entry must not be null");
                }
                out.add(hook);
            }
        }
        if (onPredictStart != null) {
            for (Consumer<PredictContext> callback : onPredictStart) {
                out.add(onPredictStart(callback));
            }
        }
        if (onPredictEnd != null) {
            for (Consumer<PredictContext> callback : onPredictEnd) {
                out.add(onPredictEnd(callback));
            }
        }
        return List.copyOf(out);
    }

    /**
     * The effective hook list for one call: defaults, then installed, then per-call.
     *
     * <p>The defaults are read HERE, at call time rather than at construction, so a hook set
     * after an agent was built still applies to it.
     */
    public static List<Hook> compose(List<? extends Hook> installed, HookCall call) {
        HookCall perCall = call == null ? HookCall.none() : call;
        List<Hook> out = new ArrayList<>(SKIP_DEFAULTS.get() ? List.of() : defaultHooks());
        if (installed != null) {
            out.addAll(installed);
        }
        out.addAll(normalise(perCall.hooks(), perCall.onPredictStart(), perCall.onPredictEnd()));
        return List.copyOf(out);
    }

    /**
     * The process-wide hooks, in order. Empty unless {@link #setDefaultHooks} was called.
     *
     * <p>A mutable COPY, as the reference returns: editing it is editing a list of your own, not
     * the process-wide one. {@link #withoutDefaultHooks} does not affect this — it suppresses the
     * defaults for a call, and a read that lied about what is installed would make a scope
     * impossible to debug from inside.
     */
    public static List<Hook> defaultHooks() {
        synchronized (DEFAULTS_MUTEX) {
            return new ArrayList<>(defaults);
        }
    }

    /**
     * Replaces the process-wide default hooks.
     *
     * <p>Defaults run before installed and per-call hooks for every agent in the process, so a
     * tracer or a metrics hook does not have to be threaded through every construction.
     */
    public static void setDefaultHooks(List<? extends Hook> hooks) {
        List<Hook> normalised = normalise(hooks, null, null);
        synchronized (DEFAULTS_MUTEX) {
            defaults = normalised;
        }
    }

    /** Appends one hook to the process-wide defaults. */
    public static void addDefaultHook(Hook hook) {
        List<Hook> normalised = normalise(List.of(hook), null, null);
        synchronized (DEFAULTS_MUTEX) {
            List<Hook> grown = new ArrayList<>(defaults);
            grown.addAll(normalised);
            defaults = List.copyOf(grown);
        }
    }

    /** Removes every process-wide default hook. */
    public static void clearDefaultHooks() {
        synchronized (DEFAULTS_MUTEX) {
            defaults = List.of();
        }
    }

    /**
     * Switches the process-wide defaults off for this thread, until the scope is closed.
     *
     * <pre>{@code
     * try (var ignored = Hooks.withoutDefaultHooks()) {
     *     agent.predict(state, questions);   // installed and per-call hooks only
     * }
     * }</pre>
     *
     * <p>Javac's {@code -Xlint:try} reports an unreferenced resource, so a build with
     * {@code -Werror} needs {@code @SuppressWarnings("try")} on the enclosing method. That is a
     * known javac wart about the idiom, not about this method.
     *
     * <p>This is the reference's {@code _SKIP_DEFAULTS}, which a router enters so that a scan
     * made of many internal predictions fires a process-wide hook once for the scan rather than
     * once per forward pass. There it is a {@code contextvars.ContextVar} and here it is a
     * {@link ThreadLocal}, and the difference is real: a {@code ContextVar} is copied into an
     * asyncio task, while a {@code ThreadLocal} is not inherited by a thread the scope starts.
     * Work handed to another thread inside the scope therefore sees the defaults again. Nothing
     * in this port dispatches hooks off the calling thread, so there is nowhere for that to bite
     * today — it is written down because the first code that does will be surprised otherwise.
     */
    public static DefaultsScope withoutDefaultHooks() {
        boolean previous = SKIP_DEFAULTS.get();
        SKIP_DEFAULTS.set(true);
        return () -> SKIP_DEFAULTS.set(previous);
    }

    /**
     * What {@link #withoutDefaultHooks} hands back.
     *
     * <p>Its own type rather than {@link AutoCloseable} so that {@code close} declares no
     * checked exception: restoring a thread-local cannot fail, and a {@code throws Exception}
     * on the resource would push a {@code catch} into every caller for a failure that does not
     * exist.
     */
    @FunctionalInterface
    public interface DefaultsScope extends AutoCloseable {
        @Override
        void close();
    }

    /**
     * Returns a positive timeout, or null for no limit.
     *
     * <p>A non-positive timeout is refused here rather than left to the wait: a zero or negative
     * deadline returns before the hook has started, so the outcome of a fast hook under one is a
     * race. The reference also refuses a NaN and an infinity, which a {@link Duration} cannot
     * hold — those two refusals are unreachable here rather than unimplemented.
     */
    public static Duration validateTimeout(Duration value) {
        if (value == null) {
            return null;
        }
        if (value.isZero() || value.isNegative()) {
            throw new IllegalArgumentException(
                    "a hook timeout must be a positive duration or null; got " + value);
        }
        return value;
    }

    /**
     * Sums the per-state usage so a hook sees one total for the call.
     *
     * <p>The reference coerces as it goes — a missing usage block, a null one, a missing key, a
     * null value and a float all become an int — because a result there is a plain dict a hook
     * may have rewritten. Here a {@link Prediction} carries a {@link com.convaiinnovations.laya.Usage}
     * whose fields are primitive ints, so every one of those coercions is unreachable. They are
     * still pinned by the fixture, which records what each degenerate shape totals to, because
     * the Java side has to produce the same number from the representable equivalent.
     */
    public static Totals aggregateUsage(List<Prediction> results) {
        int input = 0;
        int output = 0;
        for (Prediction result : results) {
            input += result.usage().inputTokens();
            output += result.usage().outputTokens();
        }
        return new Totals(input, output);
    }

    /**
     * Calls {@code event} on every hook, in order.
     *
     * <p>{@code policy.raiseErrors()} false reports through {@link Policy#onFailure} and
     * continues. {@code policy.lock()} serialises dispatch for hooks that are not re-entrant.
     * {@code policy.timeout()} bounds each hook call: an overrunning hook raises, or reports
     * when {@code raiseErrors} is false.
     *
     * <p>A timed-out hook KEEPS RUNNING. Neither Java nor Python can interrupt a thread that
     * will not cooperate, so the deadline protects the request and not the process — a hook that
     * blocks forever leaks a thread per call. That is the reference's behaviour too, and it is
     * the reason the timeout is opt-in.
     *
     * <p>{@code raiseErrors} governs a {@link RuntimeException} and nothing else. An
     * {@link Error} is rethrown whatever the policy says, which is the reference's rule in Java
     * terms: it catches {@code Exception} and deliberately not {@code BaseException}, so a
     * {@code KeyboardInterrupt} or a {@code SystemExit} is never swallowed by a telemetry hook.
     * An {@link OutOfMemoryError} reported as a line of text and then carried on from is the
     * same mistake.
     */
    public static void dispatch(List<? extends Hook> hooks, Event event, PredictContext ctx,
                                Policy policy) {
        for (Hook hook : hooks) {
            try {
                if (policy.lock() == null) {
                    call(hook, event, ctx, policy.timeout());
                } else {
                    policy.lock().lock();
                    try {
                        call(hook, event, ctx, policy.timeout());
                    } finally {
                        policy.lock().unlock();
                    }
                }
            } catch (RuntimeException problem) {
                if (policy.raiseErrors()) {
                    throw problem;
                }
                policy.onFailure().accept(String.format("laya: hook %s.%s failed: %s",
                        hook.getClass().getSimpleName(), event.wireName(), problem.getMessage()));
            }
        }
    }

    private static void call(Hook hook, Event event, PredictContext ctx, Duration timeout) {
        if (timeout == null) {
            event.callOn(hook, ctx);
            return;
        }
        Throwable[] box = new Throwable[1];
        Thread runner = new Thread(() -> {
            try {
                event.callOn(hook, ctx);
            } catch (RuntimeException | Error problem) {
                box[0] = problem;
            }
        }, "laya-hook-timeout");
        runner.setDaemon(true);
        runner.start();
        try {
            runner.join(timeout.toMillis(), timeout.toNanosPart() % 1_000_000);
        } catch (InterruptedException interrupted) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException("interrupted while waiting for hook "
                    + hook.getClass().getSimpleName() + "." + event.wireName(), interrupted);
        }
        if (runner.isAlive()) {
            // Seconds, not `Duration.toString`'s "PT0.05S". The reference names the hook as
            // `<class>.<method>` and the deadline in seconds, and a caller grepping their logs
            // across the two runtimes should find the same line.
            throw new HookTimeoutException(String.format("laya: hook %s.%s exceeded %ss",
                    hook.getClass().getSimpleName(), event.wireName(),
                    timeout.toNanos() / 1e9));
        }
        if (box[0] instanceof RuntimeException problem) {
            throw problem;
        }
        if (box[0] instanceof Error problem) {
            throw problem;
        }
    }

    /** A hook that outran {@link Policy#timeout}. Unchecked, so the usual policy governs it. */
    public static final class HookTimeoutException extends RuntimeException {

        private static final long serialVersionUID = 1L;

        HookTimeoutException(String message) {
            super(message);
        }
    }

    /**
     * Runs one call with its hooks around it: start, inference or a hook's answer, then error and
     * end.
     *
     * <p>Extracted rather than written inline in {@code Agent}, because the sequence below is the
     * whole observable contract of a hooked call and it is the same for anything that answers
     * questions. Keeping it in one place is also what lets it be tested against the reference
     * with a stub inference, which is how the fixture records it: whether the model ran at all is
     * then an observable rather than something inferred from the payload.
     *
     * <p>What happens, in order:
     *
     * <ol>
     *   <li>{@link Event#PREDICT_START} over every hook.</li>
     *   <li>If no hook assigned results, inference runs over whatever the hooks left on the
     *       context — the states, the questions and the two budget overrides. An EMPTY state list
     *       answers with an empty list without reaching inference, since there is nothing to
     *       collate.</li>
     *   <li>On a failure, {@link Event#ERROR} runs with the failure on the context. A hook that
     *       throws here is ATTACHED to the real failure as a suppressed exception, never
     *       substituted for it: the thing that broke the request must be what the caller
     *       catches.</li>
     *   <li>{@link Event#PREDICT_END} runs either way, after {@code elapsedMs} and — when there
     *       are results — the totalled {@code usage} are on the context. A hook that throws here
     *       fails the call if nothing else had, and is attached to the existing failure if
     *       something had.</li>
     * </ol>
     *
     * <p>The last two steps are written out rather than put in a {@code finally} block, and that
     * is not style: a Java {@code finally} that throws REPLACES the pending exception, which is
     * precisely the masking the reference goes out of its way to prevent.
     *
     * @return the results on the context, which is the hooks' last word on the call
     */
    public static List<Prediction> around(List<? extends Hook> hooks, PredictContext ctx,
                                          Policy policy, Inference inference) {
        Throwable raised = null;
        try {
            dispatch(hooks, Event.PREDICT_START, ctx, policy);
            if (ctx.results() == null) {
                if (ctx.states().isEmpty()) {
                    ctx.results(List.of());
                } else {
                    ctx.results(inference.run(ctx.states(), ctx.questions(), ctx.maxLen(),
                            ctx.headMaxLen()));
                }
            }
        } catch (RuntimeException | Error problem) {
            raised = problem;
            ctx.error(problem);
            try {
                dispatch(hooks, Event.ERROR, ctx, policy);
            } catch (RuntimeException | Error hookFailure) {
                // Attached, never substituted: the thing that broke the request is what the
                // caller has to catch, and a failing observer must not be able to hide it.
                problem.addSuppressed(hookFailure);
            }
        }

        ctx.elapsedMs((System.nanoTime() - ctx.startedAt()) / 1_000_000.0);
        if (ctx.results() != null) {
            ctx.usage(aggregateUsage(ctx.results()));
        }
        try {
            dispatch(hooks, Event.PREDICT_END, ctx, policy);
        } catch (RuntimeException | Error hookFailure) {
            if (raised == null) {
                throw hookFailure;
            }
            raised.addSuppressed(hookFailure);
        }

        if (raised instanceof RuntimeException problem) {
            throw problem;
        }
        if (raised instanceof Error problem) {
            throw problem;
        }
        return ctx.results();
    }
}
