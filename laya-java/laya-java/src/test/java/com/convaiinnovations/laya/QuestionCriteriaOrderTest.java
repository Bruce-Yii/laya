package com.convaiinnovations.laya;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.IdentityHashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.concurrent.ConcurrentHashMap;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/**
 * A choice's options are positional: option N's description renders Nth and logit N is read back as
 * the Nth label. So the criteria map's iteration order decides what the model was asked and how its
 * answer is labelled, and a map whose type does not define that order silently changes both.
 *
 * <p>How silent it is, measured on JDK 17:
 *
 * <pre>
 * insertion [refund, escalate, ignore]            HashMap  [ignore, escalate, refund]
 * insertion [billing, technical, sales, other]    HashMap  [other, technical, sales, billing]
 * </pre>
 *
 * Exactly reversed, so {@code probabilities().get("refund")} would have reported the logit that
 * belonged to {@code ignore}. {@code HashMap} is at least stable run to run, because
 * {@code String.hashCode} is specified. {@code Map.of} is not: it salts its layout per JVM, so three
 * runs of one program gave {@code [ignore, escalate, refund]}, {@code [escalate, ignore, refund]}
 * and {@code [refund, escalate, ignore]} -- the same question answering differently after a restart.
 */
class QuestionCriteriaOrderTest {

    /** The labels in the order logit N is read back as label N -- what the order actually decides. */
    private static List<String> renderedOptions(Question q) {
        return q.labels();
    }

    @Test
    @DisplayName("a LinkedHashMap keeps the caller's order, which is the contract")
    void linkedHashMapKeepsOrder() {
        Map<String, Object> c = new LinkedHashMap<>();
        c.put("refund", "money back for a duplicate charge");
        c.put("escalate", "pass it to a human");
        c.put("ignore", "no action needed");
        assertEquals(List.of("refund", "escalate", "ignore"),
                renderedOptions(Question.choice("What does the customer want?", c)));
    }

    @Test
    @DisplayName("a HashMap is refused, because its order is not the one that was written")
    void hashMapIsRefused() {
        Map<String, Object> c = new HashMap<>();
        c.put("refund", "money back for a duplicate charge");
        c.put("escalate", "pass it to a human");
        c.put("ignore", "no action needed");
        // Proof the hazard is real and not hypothetical: this is the order the model would have been
        // given, and it is the caller's reversed.
        assertEquals(List.of("ignore", "escalate", "refund"), new ArrayList<>(c.keySet()));
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> Question.choice("What does the customer want?", c));
        assertTrue(e.getMessage().contains("iteration order"), e.getMessage());
        assertTrue(e.getMessage().contains("LinkedHashMap"), e.getMessage());
    }

    @Test
    @DisplayName("Map.of with two or more options is refused: its order is salted per JVM")
    void mapOfIsRefused() {
        Map<String, String> c = Map.of("refund", "money back", "escalate", "to a human");
        assertThrows(IllegalArgumentException.class, () -> Question.choice("Which?", c));
    }

    @Test
    @DisplayName("Map.copyOf is refused too: copying a LinkedHashMap does not keep its order")
    void mapCopyOfIsRefused() {
        Map<String, Object> ordered = new LinkedHashMap<>();
        ordered.put("refund", "money back");
        ordered.put("escalate", "to a human");
        // Map.copyOf returns the same salted implementation Map.of does, so the order is gone.
        assertThrows(IllegalArgumentException.class,
                () -> Question.choice("Which?", Map.copyOf(ordered)));
    }

    @Test
    @DisplayName("the other unordered JDK maps are refused on the same ground")
    void otherUnorderedMapsAreRefused() {
        Map<String, Object> chm = new ConcurrentHashMap<>();
        chm.put("a", "1");
        chm.put("b", "2");
        assertThrows(IllegalArgumentException.class, () -> Question.choice("Which?", chm));
        Map<String, Object> ihm = new IdentityHashMap<>();
        ihm.put("a", "1");
        ihm.put("b", "2");
        assertThrows(IllegalArgumentException.class, () -> Question.choice("Which?", ihm));
    }

    @Test
    @DisplayName("a single option is accepted from any map: one option cannot be misordered")
    void oneOptionIsAlwaysFine() {
        assertEquals(List.of("only"),
                renderedOptions(Question.choice("Which?", Map.of("only", "the only option"))));
        assertEquals(List.of("only"),
                renderedOptions(Question.choice("Which?", new HashMap<>(Map.of("only", "d")))));
    }

    @Test
    @DisplayName("a SortedMap is accepted: its contract defines the order, even if it is not insertion")
    void sortedMapIsAccepted() {
        Map<String, Object> c = new TreeMap<>();
        c.put("refund", "money back");
        c.put("escalate", "to a human");
        assertEquals(List.of("escalate", "refund"),
                renderedOptions(Question.choice("Which?", c)));
    }

    @Test
    @DisplayName("an unmodifiable view of a LinkedHashMap is accepted: it delegates iteration")
    void unmodifiableViewIsAccepted() {
        Map<String, Object> inner = new LinkedHashMap<>();
        inner.put("refund", "money back");
        inner.put("escalate", "to a human");
        assertEquals(List.of("refund", "escalate"),
                renderedOptions(Question.choice("Which?", Collections.unmodifiableMap(inner))));
    }

    @Test
    @DisplayName("the ordered-entry factory needs no map at all, so order cannot be lost")
    void orderedEntriesFactory() {
        Question q = Question.choiceOf("What does the customer want?",
                Map.entry("refund", "money back for a duplicate charge"),
                Map.entry("escalate", "pass it to a human"),
                Map.entry("ignore", "no action needed"));
        assertEquals(List.of("refund", "escalate", "ignore"), renderedOptions(q));
    }

    @Test
    @DisplayName("choiceOf refuses a duplicate label rather than silently dropping an option")
    void orderedEntriesRefuseDuplicates() {
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> Question.choiceOf("Which?", Map.entry("a", "1"), Map.entry("a", "2")));
        assertTrue(e.getMessage().contains("duplicate"), e.getMessage());
    }
}
