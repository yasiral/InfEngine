"""Rebuild eight editorial SVG copies from verified captured data, without GPU work."""
from pathlib import Path
import argparse, csv, html, json, statistics
from verify_and_analyze import EVIDENCE, REPO
OUT = REPO/'build/evidence'

def source_data(name):
    return OUT/'ab_medians_recomputed.csv' if name=='ab_medians.csv' else EVIDENCE/'data'/name

BLUE='#2563a6'; ORANGE='#c65d24'; GREEN='#168477'; INK='#182c3d'
def rows(name): return list(csv.DictReader(source_data(name).open(encoding='utf-8')))
class Figure:
    def __init__(self,title,subtitle,height=560):
        self.height=height
        self.parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="{height}" viewBox="0 0 1000 {height}" role="img">',f'<title>{html.escape(title)}</title>',f'<desc>{html.escape(subtitle)}</desc>',f'<rect width="1000" height="{height}" fill="#fff"/>']
        self.text(35,35,title,23,bold=True); self.text(35,62,subtitle,14)
    def text(self,x,y,text,size=14,color=INK,anchor='start',bold=False):
        self.parts.append(f'<text x="{x}" y="{y}" font-family="Arial,sans-serif" font-size="{size}" fill="{color}" text-anchor="{anchor}" font-weight="{700 if bold else 400}">{html.escape(str(text))}</text>')
    def line(self,x1,y1,x2,y2,color='#dce4eb',width=1):
        self.parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}"/>')
    def rect(self,x,y,w,h,color): self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{color}"/>')
    def dot(self,x,y,color=INK,r=4): self.parts.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{color}" stroke="#fff"/>')
    def axis(self,x,y,w,h,maximum,unit):
        self.text(x,y-15,unit,15,bold=True)
        for i in range(5):
            val=maximum*i/4; yy=y+h-h*i/4
            self.line(x,yy,x+w,yy); self.text(x-8,yy+5,f'{val:g}',12,anchor='end')
        self.line(x,y,x,y+h,INK); return lambda v:y+h-v/maximum*h
    def save(self,name,footer):
        self.text(35,self.height-34,footer,12)
        self.text(35,self.height-15,'A100 80GB PCIe · Qwen2.5-7B-Instruct BF16 · patched vLLM 0.28.0 · captured evidence',12)
        self.parts.append('</svg>'); (OUT/'figures'/name).write_text('\n'.join(self.parts),encoding='utf-8')

def main():
    global OUT
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,default=OUT)
    args=ap.parse_args(); OUT=args.output.resolve()
    if OUT.is_relative_to(EVIDENCE):raise ValueError('Output must be outside immutable evidence.')
    (OUT/'figures').mkdir(parents=True,exist_ok=True)
    v=json.loads((OUT/'analysis.json').read_text(encoding='utf-8'))
    pilots=rows('pilot_profiles.csv'); med={r['experiment']:r for r in rows('ab_medians.csv')}
    # 1/2: actual offered-load ticks; two-repeat medians, cache ON.
    for num,metric,title,unit,maximum in [(1,'req_s','Pilot request throughput across offered concurrency','Completed requests / second',24),(2,'ttft_p95_s','Pilot TTFT p95 across offered concurrency','TTFT p95 (seconds)',3.5)]:
        f=Figure(title,'Prefix cache ON · CUDA graphs ON · two repeats per offered load · 15 s measurement')
        yp=f.axis(85,125,860,290,maximum,unit)
        xp=lambda n:85+(n-1)/63*860
        for profile,color in [('interactive',BLUE),('prefill',ORANGE),('mixed',GREEN)]:
            r=[r for r in pilots if r['profile']==profile]
            for a,b in zip(r,r[1:]): f.line(xp(float(a['offered_concurrency'])),yp(float(a[metric])),xp(float(b['offered_concurrency'])),yp(float(b[metric])),color,3)
            for a in r: f.dot(xp(float(a['offered_concurrency'])),yp(float(a[metric])),color)
        for n in [1,4,16,32,64]: f.text(xp(n),437,n,13,anchor='middle')
        f.text(500,462,'Offered closed-loop concurrency (clients)',15,anchor='middle')
        for x,label,col in [(85,'Interactive: 128 in / 256 out',BLUE),(385,'Prefill: 2048 in / 64 out',ORANGE),(685,'Mixed: 128 or 2048 in / 256 out',GREEN)]: f.text(x,488,label,13,col)
        f.save(f'fig{num}_'+('throughput_vs_concurrency.svg' if num==1 else 'ttft_p95_vs_concurrency.svg'),'Source: pilot_profiles.csv · short sweep selects load regions; it does not establish a GPU saturation knee.')
    # 3: categorical offered-load groups in every panel.
    f=Figure('Token-budget throughput and TTFT trade-offs','Prefill · prefix cache OFF · 8192 tokens (blue) versus 1024 (orange) · medians of three pairs',610)
    for i,(metric,unit,maximum) in enumerate([('req_s','Completed req/s',4),('ttft_p95','TTFT p95 (s)',24),('e2e_p95','E2E p95 (s)',36)]):
        x=65+i*320; yp=f.axis(x,135,265,300,maximum,unit)
        for group,exp in enumerate(['runtime_max_scheduled_tokens','runtime_max_scheduled_tokens_medium']):
            for j,arm in enumerate('AB'):
                value=float(med[exp][metric+'_'+arm]); xx=x+20+group*130+j*47
                f.rect(xx,yp(value),38,yp(0)-yp(value),[BLUE,ORANGE][j]); f.text(xx+19,yp(value)-7,f'{value:.3g}',13,anchor='middle')
            f.text(x+63+group*130,458,['Offered 64','Offered 16'][group],13,anchor='middle')
    f.text(65,501,'The budget is per scheduler step; offered load is the client concurrency, not an engine bound.',14)
    f.text(65,525,'Throughput gains at offered 64 accompany slower first tokens; the directions change at offered 16.',14)
    f.save('fig3_token_budget_high_medium.svg','Source: ab_medians.csv and ab_per_arm.csv · 30 s windows · CUDA graphs ON.')
    # 4: all configurations explicit; individual block dots around means.
    f=Figure('Concurrency and token-budget factorial interaction','Prefill · offered 96 · prefix cache OFF · three interleaved blocks · provisioned max-num-seqs 256',650)
    cells=v['factorial']['cells']
    for panel,metric,maximum,label in [(0,'req_s',4,'Completed req/s'),(1,'ttft_p95',48,'Mean of block TTFT p95 estimates (s)')]:
        x=70+panel*500; yp=f.axis(x,135,385,300,maximum,label)
        for i,c in enumerate('ABCD'):
            xx=x+25+i*91; value=cells[c][metric+'_mean']
            f.rect(xx,yp(value),50,yp(0)-yp(value),BLUE if c in 'AB' else ORANGE)
            for j,r in enumerate([r for r in v['factorial_rows'] if r['cell']==c]): f.dot(xx+15+j*10,yp(r[metric]))
            f.text(xx+25,yp(value)-10,f'{value:.4f}' if panel==0 else f'{value:.2f}',13,anchor='middle')
            f.text(xx+25,459,c,15,anchor='middle',bold=True)
            f.text(xx+25,478,f"{cells[c]['tok']} tokens",12,anchor='middle'); f.text(xx+25,495,f"Bound {cells[c]['conc']}",12,anchor='middle')
    f.text(70,535,'Dots: the three captured blocks. Bars: their means. No confidence intervals are implied.',14)
    f.text(70,563,f"Interaction (B − A) − (D − C) = {v['factorial']['effects']['interaction_abs_req_s']:+.6f} req/s",19,bold=True)
    f.save('fig4_2x2_interaction.svg','Source: 2x2_per_block.csv, reconstructed from twelve raw records · 60 s measurement per cell.')
    # 5: native counts are cumulative. Adjacent differences form disjoint bins.
    f=Figure('Native latency histogram windows and bucket resolution','Prefix cache OFF · pair 1 · each histogram covers its own final 5 s native window',740)
    for panel,(exp,family,title,arms) in enumerate([('runtime_concurrency','ttft','TTFT: concurrency bound 64 versus 16','A: 64 / B: 16'),('runtime_max_scheduled_tokens','e2e','E2E: token budget 8192 versus 1024','A: 8192 / B: 1024')]):
        y=155+panel*270; f.text(75,y-45,title,18,bold=True); yp=f.axis(75,y,870,130,1,'Fraction in disjoint native bin')
        for arm,color in [('A',BLUE),('B',ORANGE)]:
            h=v['histogram_inputs'][exp][arm]['histograms']['families'][family]; counts=h['counts']
            for i,c in enumerate(counts):
                mass=(c-(counts[i-1] if i else 0))/h['total']; assert mass>=0
                x=75+i*870/len(counts)+(0 if arm=='A' else 16)
                f.rect(x,yp(mass),15,yp(0)-yp(mass),color)
                bound=h['le'][i]; label=f'{bound:g}' if isinstance(bound,(int,float)) else str(bound)
                f.parts.append(f'<text transform="translate({x+6},{y+148}) rotate(-55)" font-family="Arial" font-size="11" fill="{INK}">{html.escape(label)}</text>') if arm=='A' else None
        hs=v['histogram_inputs'][exp]
        f.text(75,y+205,f"{arms} · blue A n={hs['A']['histograms']['families'][family]['total']}; orange B n={hs['B']['histograms']['families'][family]['total']}",13)
    f.text(75,665,'Bucket upper bounds in seconds; equally spaced categories, not a logarithmic time axis.',13)
    f.save('fig5_latency_histograms.svg','Source: archived pair-1 result.json histogram families · cumulative counts differenced between adjacent bounds.')
    # 6: explicit A/B bars and measured window medians, no missing-value zero fill.
    f=Figure('Scheduler running and waiting request counts','Prefix cache OFF · median across three arm-level window summaries',620)
    yp=f.axis(70,130,880,300,64,'Requests: blue running / green waiting')
    exps=['runtime_concurrency','runtime_concurrency_medium','runtime_max_scheduled_tokens','runtime_max_scheduled_tokens_medium']
    labels=[['Mixed / offered 64','Concurrency 64 → 16'],['Mixed / offered 16','Concurrency 16 → 4'],['Prefill / offered 64','Tokens 8192 → 1024'],['Prefill / offered 16','Tokens 8192 → 1024']]
    for i,exp in enumerate(exps):
        base=80+i*220
        for j,(arm,metric,color) in enumerate([('A','running',BLUE),('A','waiting',GREEN),('B','running',BLUE),('B','waiting',GREEN)]):
            value=float(med[exp][metric+'_'+arm]); x=base+j*44
            f.rect(x,yp(value),33,yp(0)-yp(value),color); f.text(x+16,yp(value)-8,f'{value:g}',13,anchor='middle'); f.text(x+16,452,arm+('/run' if metric=='running' else '/wait'),11,anchor='middle')
        for dy,line in enumerate(labels[i]): f.text(base+85,485+dy*22,line,13,anchor='middle')
    f.text(70,548,'These are summarized gauges, not per-request traces or a proof of an internal causal mechanism.',14)
    f.save('fig6_scheduler_running_waiting.svg','Source: ab_medians.csv and per-arm raw normalized metrics · all plotted gauges are available.')
    # 7: nine pairs displayed completely, order explicit, token hit fractions.
    f=Figure('Prefix-cache contamination across paired arms','First cache-ON pass · blue first arm / orange second arm · token hits divided by token queries',620)
    yp=f.axis(70,135,880,295,1,'Prefix-cache hit fraction (tokens)')
    cache=rows('cache_confound.csv')
    for i,r in enumerate(cache):
        base=80+i*97
        for j,(column,color) in enumerate([('first_arm_cache_frac',BLUE),('second_arm_cache_frac',ORANGE)]):
            value=float(r[column]); x=base+j*31
            f.rect(x,yp(value),26,yp(0)-yp(value),color); f.text(x+13,yp(value)-8,f'{value:.3f}',11,anchor='middle')
        f.text(base+28,453,f"Pair {r['pair']}",12,anchor='middle'); f.text(base+28,474,r['order'].replace('->','→'),12,anchor='middle')
    for x,label in [(210,'Concurrency'),(501,'Scheduled tokens'),(792,'Long-prefill threshold')]: f.text(x,508,label,15,anchor='middle',bold=True)
    f.text(70,547,f"First arms: {min(float(r['first_arm_cache_frac']) for r in cache):.3f}–{max(float(r['first_arm_cache_frac']) for r in cache):.3f}. Second arms: {min(float(r['second_arm_cache_frac']) for r in cache):.3f}–{max(float(r['second_arm_cache_frac']) for r in cache):.3f}. Cache state follows arm order.",14)
    f.save('fig7_cache_contamination.svg','Source: cache_confound.csv; checked against 18 archived before/after native counter snapshots.')
    # 8: native smoke facts, API PID precisely named.
    f=Figure('Live concurrency control, readback and restoration',f"A100 deployment smoke · native values {' → '.join(map(str,v['smoke']['native_values']))} · API PID {v['smoke']['api_pid']}",600)
    steps=[('Inspect','Native running bound: 64'),('Set concurrency to 16','CONVERGED; native running bound: 16'),('Continue inference',f"Successful completion; API PID {v['smoke']['api_pid']}"),('Restore concurrency to 64','CONVERGED; native running bound: 64'),('Inspect final identity',f"API-server PID: {v['smoke']['api_pid']} (unchanged)")]
    for i,(action,fact) in enumerate(steps):
        y=105+i*82; f.rect(60,y,350,57,'#edf4fb'); f.rect(425,y,515,57,'#f2f6f8'); f.text(77,y+35,action,16,bold=True); f.text(442,y+35,fact,16)
        if i<4: f.line(235,y+57,235,y+82,BLUE,2)
    f.text(60,543,'Readback and continuity establish functional acceptance; they do not establish a performance gain.',14)
    f.save('fig8_live_control_readback_pid.svg','Source: inputs/.../deployment/smoke/smoke.json · API PID is not an EngineCore PID.')
    print('Rebuilt eight SVGs under '+str(OUT/'figures')+'; published snapshots remain unchanged.')

if __name__=='__main__': main()
