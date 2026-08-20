// swe.w -- 2D shallow-water cell kernel, one PE per grid point.
//
// Reproduces the time step of references/shallow-water/swe.py: linearized momentum with a
// semi-implicit rotation corrector, and a fully nonlinear upwind continuity equation on an
// Arakawa C-grid. This PE owns cell (i,j); u is on its EAST face, v on its NORTH face,
// eta at its centre.
//
//   1. momentum predictor from the eta gradient
//   2. rotation corrector, using the OLD u and v
//   3. closed-basin walls: u=0 on the east face of the last column, v=0 on the north
//      face of the last row -- applied AFTER the corrector, as swe.py does
//   4. upwind face heights from the NEW velocities, strict > 0
//   5. flux divergence, one-sided at the west and south walls (the moat's zeros do that)
//   6. continuity
//
// Every constant arrives via miset from init.py rather than appearing as an :sp literal.
// Tutorials 27 and 28 inline their physics and mirror it by hand in ref.py with nothing
// checking the two copies agree; here every constant depends on the grid size anyway, and
// loading them means the kernel and mirror.py cannot disagree about a bit pattern.
//
// There is no divide and no exp: Tungsten lowers `/` to a Newton-Raphson reciprocal
// rather than an IEEE divide, and has no transcendentals at all. So 1/(1+beta_c), dt/dx,
// dt/dy and the Gaussian initial condition are all evaluated on the host.

xp param NSTEP = 1;

socket a;    // sends right (+x), receives from left  (-x)
socket b;    // sends left  (-x), receives from right (+x)
socket c;    // sends up    (+y), receives from below (-y)
socket d;    // sends down  (-y), receives from above (+y)

// State: read in by miset, written back at the end for miget.
sp eta[n=1];
sp u[n=1];
sp v[n=1];

// Per-PE constants. alpha and beta_c vary with y only; the masks are 0 on the east/north
// wall and 1 elsewhere; the rest are uniform but still loaded, not literal.
sp alpha[n=1];
sp beta_c[n=1];
sp inv1pb[n=1];
sp mask_u[n=1];
sp mask_v[n=1];
sp gdtdx[n=1];
sp gdtdy[n=1];
sp dtdx[n=1];
sp dtdy[n=1];
sp depth[n=1];

// Diagnostic taps, holding the LAST step's intermediates. 12 sp = 24 words per PE against
// 24576, so they are cheap enough to keep permanently, and they are what makes a tier-1
// mismatch a one-run diagnosis instead of a bisection: check.py can name the first
// quantity that diverges rather than only the eta it produced.
sp probe_eE[n=1]; sp probe_eW[n=1]; sp probe_eN[n=1]; sp probe_eS[n=1];
sp probe_un2[n=1]; sp probe_vn2[n=1]; sp probe_uW[n=1]; sp probe_vS[n=1];
sp probe_fe[n=1]; sp probe_fw[n=1]; sp probe_fn[n=1]; sp probe_fs[n=1];

{
    sp e; sp uu; sp vv;
    sp al; sp bc; sp ib; sp mu; sp mv;
    sp gx; sp gy; sp tx; sp ty; sp dep;

    e   <- eta[0];
    uu  <- u[0];
    vv  <- v[0];
    al  <- alpha[0];
    bc  <- beta_c[0];
    ib  <- inv1pb[0];
    mu  <- mask_u[0];
    mv  <- mask_v[0];
    gx  <- gdtdx[0];
    gy  <- gdtdy[0];
    tx  <- dtdx[0];
    ty  <- dtdy[0];
    dep <- depth[0];

    j in [0, NSTEP) {
        sp eW[1]; sp eE[1]; sp eS[1]; sp eN[1];

        // PHASE 1 -- eta to all four neighbours.
        // The sends and receives MUST share one `parallel` block. Split into two serial
        // blocks and every PE blocks on its first send waiting for a reader that is
        // itself still sending: the whole grid deadlocks with no diagnostic. 8 concurrent
        // operations, against a ceiling of about 13 hardware threads.
        parallel {
            a[] <- e;
            b[] <- e;
            c[] <- e;
            d[] <- e;
            eW[0] <- a[];
            eE[0] <- b[];
            eS[0] <- c[];
            eN[0] <- d[];
        }

        sp un; sp vn; sp un2; sp vn2;

        // Momentum predictor. On the east wall eE is the moat's zero, so `un` is garbage
        // there; the wall assignment below discards it, which is exactly what swe.py does
        // when it overwrites u_np1[-1,:] after the corrector.
        un <- uu - gx * (eE[0] - e);
        vn <- vv - gy * (eN[0] - e);

        // Rotation corrector. Reads the OLD uu and vv, so both new velocities are formed
        // before either is committed -- committing uu early would feed the v corrector
        // the new value and silently change the answer.
        un2 <- (un - bc * uu + al * vv) * ib;
        vn2 <- (vn - bc * vv - al * uu) * ib;

        // Closed-basin walls, applied AFTER the corrector (swe.py:203-204). mu and mv are
        // 0 on the east/north wall and 1 elsewhere.
        //
        // Arithmetic rather than predicated, for the codegen reason documented at the flux
        // split below. `* mu` on its own would give -0.0 for a negative un2, and that sign
        // survives into the next step's `- al * uu` wherever the other operand is also
        // zero -- at NX=32 that is 312 of 1024 cells at step 1. Adding +0.0 collapses it
        // to the exact +0.0 that swe.py's `u_np1[-1,:] = 0.0` assigns, and leaves every
        // non-wall value untouched, since `x * 1.0 + 0.0 == x`.
        un2 <- un2 * mu + 0.0:sp;
        vn2 <- vn2 * mv + 0.0:sp;

        sp uW[1]; sp uE[1]; sp vS[1]; sp vN[1];

        // PHASE 2 -- the new velocities: u horizontally, v vertically.
        //
        // The leftward and downward sends look redundant but are load-bearing, and so are
        // the receives of uE and vN, which the scheme never uses. Both were checked by
        // ablation at 4x4 and both corrupt the whole grid rather than hanging cleanly:
        // dropping the b/d sends gives 7.6e-04 error in eta, dropping the uE/vN receives
        // gives 1.6e-04 and 129 orphan-wavelet reports.
        //
        // Mechanism: the moat is purely reactive, so with no `b` send the west moat never
        // fires and the x=0 column never receives its uW. Dropping a receive instead
        // leaves a wavelet in the queue, so the next step's eE reads it and every socket
        // is permanently one wavelet out of phase. simfabric stops on quiescence rather
        // than hanging, so either way you get a completed run with wrong numbers.
        parallel {
            a[] <- un2;
            b[] <- un2;
            c[] <- vn2;
            d[] <- vn2;
            uW[0] <- a[];
            uE[0] <- b[];
            vS[0] <- c[];
            vN[0] <- d[];
        }

        // Total column height h = H + eta, here and at the four neighbours.
        sp ec; sp eEc; sp eWc; sp eNc; sp eSc;
        ec  <- e     + dep;
        eEc <- eE[0] + dep;
        eWc <- eW[0] + dep;
        eNc <- eN[0] + dep;
        eSc <- eS[0] + dep;

        // Upwind face fluxes, written branch-free as a flux split:
        //
        //     u * h_upwind  ==  max(u,0) * h_here + min(u,0) * h_downwind
        //
        // which is exactly swe.py's `u * where(u > 0, eta_here + H, eta_downwind + H)`,
        // because one of the two terms is always a zero product. It also reproduces the
        // strict `> 0` test: at u == 0 both terms vanish, which is what multiplying the
        // downwind height by a zero velocity gives.
        //
        // This is not a stylistic preference. The predicated form --
        //     sp h; h <- downwind;  if (u > 0.0:sp) { h <- upwind; }
        // -- MISCOMPILES on arch=sdr: `h <- downwind` is a plain copy, register allocation
        // coalesces h with its source, and the predicated select is emitted with identical
        // source registers, so the body never executes and every upwind cell silently takes
        // the downwind value. The listing shows the tell:
        //     flteqs P0 = D0, 0x4;  P0? select32 D0 = D0, D0, P0;
        // It cost the first two runs of this kernel, presenting as a 7e-9 discrepancy in
        // eta confined to two walls -- and it is why the wall masks above were fine while
        // these were not: un2 holds an expression result, not a copy.
        //
        // if/else and the ternary both work. This avoids predication entirely anyway,
        // because the trigger is a register-allocation decision rather than anything
        // visible in the source. See ../probe-select/ for the measured matrix.
        sp uep; sp uen; sp uwp; sp uwn;
        sp vnp; sp vnn; sp vsp; sp vsn;
        uep <- max(un2,   0.0:sp);  uen <- min(un2,   0.0:sp);
        uwp <- max(uW[0], 0.0:sp);  uwn <- min(uW[0], 0.0:sp);
        vnp <- max(vn2,   0.0:sp);  vnn <- min(vn2,   0.0:sp);
        vsp <- max(vS[0], 0.0:sp);  vsn <- min(vS[0], 0.0:sp);

        sp fe; sp fw; sp fn; sp fs;
        fe <- uep * ec  + uen * eEc;   // east  face: upwind cell is this one
        fw <- uwp * eWc + uwn * ec;    // west  face: upwind cell is the west neighbour
        fn <- vnp * ec  + vnn * eNc;   // north face
        fs <- vsp * eSc + vsn * ec;    // south face

        // Continuity. At the west wall fw is zero because the moat's reply makes both uwp
        // and uwn zero, which is swe.py's one-sided uhwe[0,:] = u_np1[0,:]*h_e[0,:]. PE i's
        // `fe` and PE i+1's `fw` are the same expression on bit-identical operands, so the
        // flux telescopes and sum(eta) is conserved to roundoff.
        sp en;
        en <- e - (tx * (fe - fw) + ty * (fn - fs));

        probe_eE[0] <- eE[0]; probe_eW[0] <- eW[0];
        probe_eN[0] <- eN[0]; probe_eS[0] <- eS[0];
        probe_un2[0] <- un2;  probe_vn2[0] <- vn2;
        probe_uW[0] <- uW[0]; probe_vS[0] <- vS[0];
        probe_fe[0] <- fe;    probe_fw[0] <- fw;
        probe_fn[0] <- fn;    probe_fs[0] <- fs;

        // Commit last, all three together: `e` is still needed by every face height above.
        uu <- un2;
        vv <- vn2;
        e  <- en;
    }

    // Write back, or miget returns the initial state.
    eta[0] <- e;
    u[0]   <- uu;
    v[0]   <- vv;
}
