"""AIGP tune diagnostic protocol — analytical, repeatable.
Ingests a filt*.csv, isolates each gate approach, and reports the closest-approach
error + the prescribed single-parameter adjustment. No eyeballing."""
import csv, sys

def load(p):
    R=[]
    for r in csv.DictReader(open(p)):
        d={}
        for k,v in r.items():
            try: d[k]=float(v)
            except: d[k]=v
        R.append(d)
    return R

def gate_windows(R):
    """A gate approach = contiguous ticks with a real detection (sz>=0.05) building to a
    local size peak, then a collapse (>40% drop) or loss. Return list of (i0,ipeak,i1)."""
    wins=[]; i=0; n=len(R)
    while i<n:
        if R[i].get('size_frac',0) and R[i]['size_frac']>=0.05:
            j=i; peak=i
            while j<n and (R[j].get('size_frac',0)>=0.03):
                if R[j]['size_frac']>R[peak]['size_frac']: peak=j
                # stop the window once size has collapsed to <45% of peak AFTER the peak
                if j>peak and R[peak]['size_frac']>=0.15 and R[j]['size_frac']<0.45*R[peak]['size_frac']:
                    break
                j+=1
            if R[peak]['size_frac']>=0.15:   # only count windows that got a real near gate
                wins.append((i,peak,j))
            i=j+1
        else:
            i+=1
    return wins

def rail_report(R):
    """PER-LEG rail health -- the LOCK RATE, which is the number the rail rebuild is
    judged on.

    AREA IS NO LONGER A VALIDITY TEST and is not reported as one. The old detector
    masked cyan and thresholded its area; that is the wrong question about two thin
    bright LINES, because a perfectly visible rail one pixel wide has ~zero area. The
    flt9 gate-3 leg read 'found on 8% of ticks, area p50 0.0000' -- a FALSE NEGATIVE.
    The rebuilt detector fits the two rail curves and locks on fit quality + rail
    separation + row count, so the questions now are:

    1. LOCK RATE per leg. flt9's gate-3 leg (ag=2) is the baseline to beat: 7.7% found,
       3.4% 'solid' under the old area gate.
    2. WHEN IT DOES NOT LOCK, WHY? rows/rms/sep separate 'the path is not in frame'
       (few rows) from 'something is in frame but is not a pair of rails' (bad rms).
    """
    legs = sorted({int(r['active_gate']) for r in R if isinstance(r.get('active_gate'), float)})
    if not legs:
        return
    has_new = any('fit_rms' in r for r in R)
    print("  ---- RAIL lock per leg (the rebuild's headline number) ----")
    if has_new:
        print("  leg  ticks   lock%   |u_tube| p50  p90    v_conv p50   sep p50   rms p50   rows p50")
    else:
        print("  leg  ticks  found%   area p50   area p90   |u_tube| max   [OLD detector log]")
    for leg in legs:
        W=[r for r in R if int(r.get('active_gate',-1))==leg]
        if not W: continue
        fnd=[r for r in W if r.get('tube_found')]
        pct=lambda v,q: (sorted(v)[min(len(v)-1,int(q*len(v)))] if v else 0.0)
        if has_new:
            lk=[r for r in W if r.get('tube_solid')]
            u=[abs(r.get('u_tube',0.0)) for r in lk]
            print(f"   {leg}   {len(W):5d}   {100*len(lk)/len(W):5.1f}"
                  f"   {pct(u,0.50):10.3f}  {pct(u,0.90):5.3f}"
                  f"   {pct([r.get('v_converge',0.0) for r in lk],0.50):+10.3f}"
                  f"   {pct([r.get('rail_sep',0.0) for r in lk],0.50):7.3f}"
                  f"   {pct([r.get('fit_rms',0.0) for r in lk],0.50):7.2f}"
                  f"   {pct([r.get('rail_rows',0.0) for r in lk],0.50):8.0f}")
        else:
            ar=sorted(r.get('area_frac',0.0) for r in W)
            p=lambda q: ar[min(len(ar)-1,int(q*len(ar)))]
            umax=max((abs(r.get('u_tube',0.0)) for r in fnd), default=0.0)
            print(f"   {leg}   {len(W):5d}   {100*len(fnd)/len(W):5.1f}"
                  f"   {p(0.50):8.4f}   {p(0.90):8.4f}   {umax:11.3f}")
    # Command saturation per leg -- the flt9 defect was des_roll pinned to its clamp.
    if any('des_roll_deg' in r for r in R):
        print("  ---- lateral command saturation (flt9 gate-3 leg: 65% pinned at -11 deg) ----")
        for leg in legs:
            W=[r for r in R if int(r.get('active_gate',-1))==leg
               and isinstance(r.get('des_roll_deg'), float)]
            if not W: continue
            pk=max(abs(r['des_roll_deg']) for r in W)
            if pk < 1e-6: continue
            sat=sum(1 for r in W if abs(r['des_roll_deg'])>pk-0.05)
            com=""
            if any('commit_k' in r for r in W):
                n=sum(1 for r in W if r.get('committed'))
                mk=min((r.get('commit_k',1.0) for r in W), default=1.0)
                com=(f"   COMMIT fired on {n:3d} ticks, min commit_k {mk:.3f}"
                     f"{'  <-- MUST BE 0 / 1.000 on legs 0-1' if leg<2 else ''}")
            print(f"   leg {leg}: peak |des_roll| {pk:5.1f}deg   at-peak {100*sat/len(W):4.1f}%"
                  f" of {len(W)} ticks{com}")
    # The ladder itself, if the flight logged it.
    if any('descent_bias' in r for r in R):
        print("  ---- altitude ladder (active bias per leg) ----")
        for leg in legs:
            W=[r for r in R if int(r.get('active_gate',-1))==leg]
            b={round(r.get('descent_bias',0.0),4) for r in W}
            print(f"   leg {leg}: descent_bias {sorted(b)}  base_thrust "
                  f"{min(r.get('base_thrust',0.0) for r in W):.3f}"
                  f"..{max(r.get('base_thrust',0.0) for r in W):.3f}")


def analyze(path):
    R=load(path)
    wins=gate_windows(R)
    print(f"\n===== {path.split('/')[-1]} :: {len(wins)} gate approach(es) =====")
    for k,(i0,ip,i1) in enumerate(wins,1):
        w=R[i0:i1+1]
        pk=R[ip]
        ev=pk.get('v_err'); eu=pk.get('u_err'); szmax=pk.get('size_frac')
        # overshoot: sign flips of v_err across the window
        vs=[r['v_err'] for r in w if isinstance(r.get('v_err'),float)]
        flips=sum(1 for a,b in zip(vs,vs[1:]) if (a>0)!=(b>0))
        # saturation of vertical trim in window
        vtr=[r.get('vtrim',0) for r in w if isinstance(r.get('vtrim'),float)]
        up_sat=sum(1 for x in vtr if x>=0.0995)/max(len(vtr),1)
        dn_sat=sum(1 for x in vtr if x<=-0.0445)/max(len(vtr),1)
        lp=[r['loop_hz'] for r in w if isinstance(r.get('loop_hz'),float) and r['loop_hz']>0]
        print(f" gate#{k}  t={w[0]['t']:.1f}-{w[-1]['t']:.1f}  peak sz={szmax:.3f} @t={pk['t']:.2f}")
        print(f"   CLOSEST-APPROACH ERROR:  v_err={ev:+.3f} (vert)   u_err={eu:+.3f} (lat)")
        print(f"   vert overshoot (sign flips)={flips}   up-clamp {100*up_sat:.0f}%  down-clamp {100*dn_sat:.0f}%")
        print(f"   loop_hz min={min(lp) if lp else 0:.0f}")
    rail_report(R)
    return R, wins

paths=sys.argv[1:] or []
for p in paths: analyze(p)
