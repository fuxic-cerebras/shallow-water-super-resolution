// moat.w -- closed-basin wall kernel for the shallow-water grid.
//
// Adapted from tutorial 27-wave-equation's moat.w; the body is unchanged, only the
// explanation differs. Do NOT substitute tutorial 28's echoing moat -- a Neumann echo
// would return the neighbour's velocity instead of zero and break every wall.
//
// Shore tiles ring the (NX+2)x(NY+2) rect and reply 0:sp to any wavelet, on the socket
// paired with the one that received it. For this scheme that single reply value does all
// four walls at once:
//
//   west  (x=0)    u_W = 0  ->  uhwe = un2*h_e - 0*h_w, i.e. swe.py's one-sided
//                               uhwe[0,:] = u_np1[0,:]*h_e[0,:]
//                  eta_W = 0 -> never read: h_w's `u_W > 0` test is false, so h_w takes
//                               the `eta + depth` branch, which IS swe.py's h_w[0,:]
//   south (y=0)    same in v
//   east  (x=NX-1) the interior PE zeroes its own un2, so the reply only ever feeds
//                  h_e, which is then multiplied by that zero
//   north (y=NY-1) same in v
//
// Two wavelets arrive per socket per step (eta in phase 1, a velocity in phase 2) and
// `forever dispatch` is reactive and unbounded, so both are answered. Corner tiles
// receive nothing: only cardinal sends exist, and a corner's orthogonal neighbours are
// themselves moat tiles, which never initiate.

socket a;
socket b;
socket c;
socket d;

parallel {

	forever dispatch(b) {
	    wavelet() {
	        a[] <- 0:sp;
	    }
	}

	forever dispatch(a) {
	    wavelet() {
	        b[] <- 0:sp;
	    }
	}

	forever dispatch(d) {
	    wavelet() {
	        c[] <- 0:sp;
	    }
	}

	forever dispatch(c) {
	    wavelet() {
	        d[] <- 0:sp;
	    }
	}

}
