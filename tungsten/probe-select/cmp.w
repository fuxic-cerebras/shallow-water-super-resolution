// Characterize the predicated-select miscompile on arch=sdr.
//
// A bare `if` (no `else`) that conditionally overwrites a variable produces a no-op
// whenever that variable's current value is a plain COPY of another variable. Register
// allocation coalesces the destination with the copy source, so the emitted select has
// identical source registers -- `select32 Dx = Dx, Dx` -- and the assignment in the body
// never takes effect. The body is simply skipped, silently.
//
// The six cases below isolate that. `v` is the condition value; `av` and `bv` are genuine
// runtime values loaded via miset, so nothing can be constant-folded.

xp param N = 8;

sp x[n=N];
sp a[n=N];
sp b[n=N];

sp k_copy_lit[n=N];   // default = copy,       branch = literal    -> WRONG
sp k_copy_var[n=N];   // default = copy,       branch = variable   -> WRONG
sp k_lit_var[n=N];    // default = literal,    branch = variable   -> ok
sp k_expr_lit[n=N];   // default = expression, branch = literal    -> ok
sp k_ifelse[n=N];     // if/else, both variables                   -> ok
sp k_ternary[n=N];    // ?:,      both variables                   -> ok

{
    sp zero;
    zero <- 0.0:sp;

    i in [0, N) {
        sp v; sp av; sp bv;
        v  <- x[i];
        av <- a[i];
        bv <- b[i];

        sp h;

        // The two broken forms. Both have a plain copy as the fall-through value.
        h <- bv;         if (v > 0.0:sp) { h <- 1.0:sp; }  k_copy_lit[i] <- h;
        h <- bv;         if (v > 0.0:sp) { h <- av;     }  k_copy_var[i] <- h;

        // A literal default is materialized into its own register, so the select has two
        // distinct sources and behaves.
        h <- 2.0:sp;     if (v > 0.0:sp) { h <- av;     }  k_lit_var[i]  <- h;

        // So is an expression result -- `+ zero` is enough to defeat the coalescing.
        h <- bv + zero;  if (v > 0.0:sp) { h <- 1.0:sp; }  k_expr_lit[i] <- h;

        // Both working alternatives: neither leaves a fall-through copy to coalesce.
        if (v > 0.0:sp) { h <- av; } else { h <- bv; }      k_ifelse[i]   <- h;
        h <- (v > 0.0:sp) ? av : bv;                        k_ternary[i]  <- h;
    }
}
