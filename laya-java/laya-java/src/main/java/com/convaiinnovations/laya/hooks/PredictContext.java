package com.convaiinnovations.laya.hooks;

import com.convaiinnovations.laya.Prediction;
import com.convaiinnovations.laya.Predictor;
import com.convaiinnovations.laya.Question;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;

/**
 * One call, as every hook of that call sees it — and as a hook may change it.
 *
 * <p>Mutable, which nothing else in this port is. That is the point: a hook that could only read
 * would be a logger, and the reference's contract is that a start hook may rewrite the states,
 * the questions and the token budget, and that an end hook may rewrite the results. A record
 * cannot express any of that.
 *
 * <p>One context exists per call and is shared by every event of it, so {@link #runId} is what
 * lets a tracer pair a start with its end without threading state of its own. Identity is
 * identity: the reference declares {@code eq=False} so two contexts are never equal and one can
 * sit in a set keyed by the call rather than by its contents, which is what a Java object does
 * by default — so there is no {@code equals} or {@code hashCode} here, on purpose.
 *
 * <p>Not thread-safe, and not meant to be: hooks run on the calling thread, in order.
 *
 * <p>Two of the reference's fields are absent. {@code router} and {@code decision} carry a
 * {@code RouteDecision} to an {@code on_route} hook, and nothing in this port dispatches that
 * event yet — {@link com.convaiinnovations.laya.Router} does its own selection without hooks. A
 * field that is always null is not a port of a field, it is a promise the caller cannot tell
 * from a bug. What the reference splits across {@code agent} and {@code router} is one field
 * here, {@link #predictor}, because both are a {@link Predictor}.
 */
public final class PredictContext {

    private final String runId = UUID.randomUUID().toString().replace("-", "");
    private final long startedAt = System.nanoTime();
    private final String model;
    private final Predictor predictor;

    private List<Object> states;
    private Map<String, Question> questions;
    private List<Prediction> results;
    private Integer maxLen;
    private Integer headMaxLen;
    private Hooks.Totals usage;
    private Double elapsedMs;
    private Throwable error;

    /**
     * A context for one call.
     *
     * @param model      the resolved checkpoint name, or null when the caller named none
     * @param predictor  the agent or router answering this call, or null
     * @param maxLen     a per-call token budget, or null for the checkpoint's own
     * @param headMaxLen a per-call head budget, or null for the checkpoint's own
     */
    public PredictContext(List<?> states, Map<String, Question> questions, String model,
                          Predictor predictor, Integer maxLen, Integer headMaxLen) {
        this.states = new ArrayList<>(states);
        this.questions = new LinkedHashMap<>(questions);
        this.model = model;
        this.predictor = predictor;
        this.maxLen = maxLen;
        this.headMaxLen = headMaxLen;
    }

    /** A context with no budget overrides. */
    public PredictContext(List<?> states, Map<String, Question> questions, String model,
                          Predictor predictor) {
        this(states, questions, model, predictor, null, null);
    }

    /** Shared by every hook of one call, 32 hex characters, as the reference's {@code uuid4().hex}. */
    public String runId() {
        return runId;
    }

    /** The resolved checkpoint name, or null. */
    public String model() {
        return model;
    }

    /** The agent or router answering this call, or null. */
    public Predictor predictor() {
        return predictor;
    }

    /** The states this call will be answered over. */
    public List<Object> states() {
        return states;
    }

    /** Replaces the states. Inference is handed what the LAST start hook left here. */
    public void states(List<?> replacement) {
        this.states = new ArrayList<>(replacement);
    }

    /** The questions this call will ask. */
    public Map<String, Question> questions() {
        return questions;
    }

    /** Replaces the questions, keeping the given iteration order — a choice's options are positional. */
    public void questions(Map<String, Question> replacement) {
        this.questions = new LinkedHashMap<>(replacement);
    }

    /**
     * The answers, or null when the model has not run.
     *
     * <p>Null and empty are different states of a call: "inference has not happened" against
     * "inference returned nothing". A start hook reads this to tell whether an earlier hook has
     * already answered, and {@link Hooks#around} reads it to decide whether to run the model at
     * all, so collapsing the two would make an empty batch skip inference forever.
     */
    public List<Prediction> results() {
        return results;
    }

    /** Replaces the results. From an end hook this is what the caller receives. */
    public void results(List<Prediction> replacement) {
        this.results = replacement;
    }

    /** A per-call token budget, or null for the checkpoint's own. */
    public Integer maxLen() {
        return maxLen;
    }

    /** Overrides the token budget for this call. */
    public void maxLen(Integer replacement) {
        this.maxLen = replacement;
    }

    /** A per-call head budget, or null for the checkpoint's own. */
    public Integer headMaxLen() {
        return headMaxLen;
    }

    /** Overrides the head budget for this call. */
    public void headMaxLen(Integer replacement) {
        this.headMaxLen = replacement;
    }

    /** The call's totalled usage, set before the end hooks run, or null when it failed. */
    public Hooks.Totals usage() {
        return usage;
    }

    /** Set by {@link Hooks#around} once the results are known. */
    void usage(Hooks.Totals totals) {
        this.usage = totals;
    }

    /** Wall time for the whole call, set before the end hooks run. */
    public Double elapsedMs() {
        return elapsedMs;
    }

    /** {@link System#nanoTime} at construction, for a hook measuring its own slice of the call. */
    public long startedAt() {
        return startedAt;
    }

    /** Set by {@link Hooks#around} once the call is over, failed or not. */
    void elapsedMs(double value) {
        this.elapsedMs = value;
    }

    /** What the call failed with, or null. Set before the error and end hooks run. */
    public Throwable error() {
        return error;
    }

    /** Set by {@link Hooks#around} before the error hooks run. */
    void error(Throwable failure) {
        this.error = failure;
    }

    /**
     * Answers the call from a start hook: inference is skipped and the end hooks still run.
     *
     * <p>{@code results} replaces the WHOLE call, so it carries one entry per state in
     * {@link #states}, in that order — the shape {@code predictBatch} returns. A hook fires once
     * per call and a call can carry many states. The one other accepted shape is a single entry
     * for the whole call.
     *
     * <p>The count is checked HERE, against that contract, so a wrong one fails inside the hook
     * under the caller's own {@link Hooks.Policy#raiseErrors} rather than downstream. In the
     * reference, downstream was where it surfaced: a router indexed {@code results[0]} of an
     * empty list, which the server mapped to a 500, and a batch returned a shorter list than it
     * was given states — quietly dropping rows the caller was about to zip against.
     *
     * <p>Zero results for zero states is accepted, because zero is both "one per state" and
     * "nothing", and an empty call has nothing to answer.
     *
     * @throws IllegalArgumentException when the count is neither 1 nor one per state
     */
    public void skip(List<Prediction> results) {
        if (results.size() != 1 && results.size() != states.size()) {
            throw new IllegalArgumentException(String.format(
                    "ctx.skip() takes one result for the whole call or one per state in "
                    + "ctx.states (%d); got %d", states.size(), results.size()));
        }
        this.results = results;
    }

    @Override
    public String toString() {
        return "PredictContext[runId=" + runId + ", states=" + states.size()
                + ", questions=" + questions.size()
                + ", results=" + (results == null ? "none" : String.valueOf(results.size()))
                + ", model=" + model + "]";
    }
}
