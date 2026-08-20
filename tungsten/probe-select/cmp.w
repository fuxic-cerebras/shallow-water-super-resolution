// Does min/max on sp behave, including at signed zeros? The upwind flux can be written
// branch-free as u*h_upwind == max(u,0)*h_here + min(u,0)*h_downwind, which avoids the
// predicated-select path entirely -- but only if min/max are sane.
xp param N = 8;

sp x[n=N];
sp r_max[n=N];
sp r_min[n=N];
sp r_split[n=N];   // max(x,0)*A + min(x,0)*B, with A=10, B=20
sp r_where[n=N];   // x * ((x>0) ? A : B), the form being replaced

{
    sp A; sp B;
    A <- 10.0:sp;
    B <- 20.0:sp;

    i in [0, N) {
        sp v;
        v <- x[i];

        sp p; sp m;
        p <- max(v, 0.0:sp);
        m <- min(v, 0.0:sp);
        r_max[i] <- p;
        r_min[i] <- m;
        r_split[i] <- p * A + m * B;

        sp h; h <- B; if (v > 0.0:sp) { h <- A; }
        r_where[i] <- v * h;
    }
}
