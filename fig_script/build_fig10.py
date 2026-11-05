# -*- coding: utf-8 -*-
# FULL Fig 10: OVERVIEW (Afternoon hero + Morning/Evening/Pre-dawn thumbnails,
# GT campus boundary, geographic aspect, offset-from-regional labels) + DETAILS
# (i1-i10 land-cover insets with per-cover skill + per-site +/-conformal band).
import io,sys,os,json,pickle,warnings,urllib.request

def _assert_cache_orientation(_msd):
    """Tripwire (2026-08-25): cache tuples are (prediction, observation). The Persistence
    forecast must equal the observation lagged by the horizon; fails loudly if anyone
    ever swaps the unpack order again."""
    import numpy as _np
    _hmap={'5min':1,'10min':2,'1h':12,'6h':72,'24h':288}
    _k=next(k for k in _msd if k[2]=='Persistence' and k[1] in _hmap)
    _p=_np.asarray(_msd[_k][0]).ravel(); _o=_np.asarray(_msd[_k][1]).ravel(); _h=_hmap[_k[1]]
    assert _np.abs(_p[_h:]-_o[:-_h]).max()<1e-3, (
        'ms_predictions cache orientation violated: element0 must be PREDICTION, element1 OBSERVATION')

warnings.filterwarnings('ignore'); sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8',line_buffering=True)
import numpy as np, pandas as pd, matplotlib.pyplot as plt, matplotlib.colors as mcolors, matplotlib.gridspec as gridspec
import matplotlib.patheffects as pe
from matplotlib.patches import Circle, Patch, Ellipse, Polygon
from matplotlib.lines import Line2D
sys.path.insert(0,'fig_script')
from fig_config import FULL_W, STATION_COORDS, CANDIDATES, LC_COLORS, apply_rcparams, FS_LABEL, FS_TICK, FS_DENSE_SM, FS_DENSE, savefig
apply_rcparams()
LABEL_OFFSETS={10635576:(14,2),10635580:(-14,4),10635581:(14,2),10635582:(14,2),10635585:(14,2),10635587:(14,2),10635588:(14,2),10635590:(14,2),10635593:(14,2),10635595:(14,2)}   # site 3 label moved RIGHT (user 2026-08-16) so the 200m top band could shrink to 140m -> shorter figure; above-placement history: was LEFT (crossed site 4's (graphic-designer audit, MEDIUM)
ELAB={10635593:1,10635582:2,10635576:3,10635588:4,10635595:5,10635585:6,10635587:7,10635580:8,10635590:9,10635581:10}
LCZC={1:'#8c0000',2:'#d10000',3:'#ff0000',4:'#bf4d00',5:'#ff6600',6:'#ff9955',8:'#eeeeee'}
MNAME={'PatchTST':'PT','iTransformer':'iT','LSTM':'LS','Attention-LSTM':'AL'}  # 2-letter model codes (= Fig 9b badges) — ties each inset to Fig 9b's best-model column AND frees room for the per-site MAE
QH={'Morning':(6,12),'Afternoon':(12,18),'Evening':(18,24),'Pre-dawn':(0,6)}
# column-major-by-land-cover order: col0..4 = Built/Imperv/Exposed/Open green/Tree; row0=i1-5, row1=i6-10
ZOOM=[10635588,10635595,10635593,10635576,10635582,10635590,10635581,10635585,10635580,10635587]   # columns reordered cool->hot: Open green | Tree | Built | Exposed | Impervious (panel-i redesign move #1)
BASE=os.getcwd()   # run from the project root (renders fig10_landcover_forecast.png there)
_C1=os.path.join(os.path.dirname(os.path.abspath(__file__)),'fig10_dev_2026-07','build_fig10_data.pkl')   # project copy of the data cache (preferred; c:/tmp fallback)
CACHE=_C1 if os.path.exists(_C1) else 'c:/tmp/build_fig10_data.pkl'
if os.path.exists(CACHE):
    ST,Kq,LC_SKILL,bld,rg,lcz,meta,gtb=pickle.load(open(CACHE,'rb')); print('loaded cache')
else:
    import osmnx as ox
    from shapely.geometry import Point as _Pt
    RUN=BASE+'/output/TF_for-upload_4(H100)__output__completed/output/run_2026-03-01_20-51'
    ms=pickle.load(open(RUN+'/ms_predictions_cache.pkl','rb'))['ms_predictions']
    _assert_cache_orientation(ms)
    metr=pd.read_csv(RUN+'/multistation_perstation_metrics.csv'); conf=pd.read_csv(RUN+'/conformal_prediction_results.csv')
    lcz=np.load('cache/lcz_campus.npy'); meta=json.load(open('cache/lcz_campus_meta.json'))
    url=('https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?station=ATL&data=tmpc&year1=2015&month1=6&day1=5&year2=2017&month2=6&day2=28&tz=America%2FNew_York&format=onlycomma&latlon=no&missing=M&report_type=3&report_type=4')
    k=pd.read_csv(io.StringIO(urllib.request.urlopen(url,timeout=120).read().decode())); k['valid']=pd.to_datetime(k['valid']); k['tmpc']=pd.to_numeric(k['tmpc'],errors='coerce'); k=k.dropna(subset=['tmpc']).set_index('valid')
    km=k[(k.index.month==5)&(k.index.day>=15)&(k.index.day<=30)]; KH=km.groupby(km.index.hour).tmpc.mean()
    Kq={q:KH.loc[h0:h1-1].mean() for q,(h0,h1) in QH.items()}
    # per-cover skill (best model per cover) + per-station
    h6=metr[metr.Horizon=='6h']; cov={}
    for sid,sc in STATION_COORDS.items():
        for mdl in list(CANDIDATES)+['DLinear']:
            row=h6[(h6.Station==sid)&(h6.Model==mdl)]
            if len(row): cov.setdefault(sc['lc'],{}).setdefault(mdl,[]).append(row.MAE.iloc[0])
    LC_SKILL={}
    for lc,md in cov.items():
        cm={m:np.mean(v) for m,v in md.items() if m in CANDIDATES}; bm=min(cm,key=cm.get); dl=np.mean(md['DLinear'])
        LC_SKILL[lc]=(bm,cm[bm],(dl-cm[bm])/dl*100)
    bld=ox.features_from_point((33.779,-84.399),tags={'building':True},dist=1200); bld=bld[bld.geometry.geom_type.isin(['Polygon','MultiPolygon'])]
    rg=ox.graph_to_gdfs(ox.graph_from_point((33.779,-84.399),dist=1200,network_type='drive'),nodes=False)
    gtb=ox.geocode_to_gdf('Georgia Institute of Technology, Atlanta, Georgia, USA')
    ST={}
    for sid,sc in STATION_COORDS.items():
        sub=h6[(h6.Station==sid)&(h6.Model.isin(CANDIDATES))]; bm=sub.loc[sub.MAE.idxmin(),'Model']
        dlm=h6[(h6.Station==sid)&(h6.Model=='DLinear')].MAE.iloc[0]; sk=(dlm-sub.MAE.min())/dlm*100  # per-station skill vs DLinear (full-test, paper basis)
        yp,yt=ms[(str(sid),'6h',bm)]; n=len(yt)   # cache stores (preds, actuals) - orientation fixed 2026-08-25
        df=pd.read_csv('00_dataset/5min/20 stations/%d_5min_Interval.csv'%sid,low_memory=False,parse_dates=['Datetime']).dropna(subset=['Temp','RH','DewPt']).sort_values('Datetime').set_index('Datetime').loc['2015-06-05':'2017-06-28']
        te=df.iloc[int(len(df)*0.85):]; tt=te['Temp'].values; bo,bc=0,-1; chk=min(500,n)
        for off in range(0,min(1000,len(tt)-n)+1):   # + 1: the true offset len - n was excluded, labels sat 5 min early (2026-09-30)
            c=np.corrcoef(yt[:chk],tt[off:off+chk])[0,1]
            if c>bc: bc,bo=c,off
        ts=te.index[bo:bo+n]; mask=(ts>=pd.Timestamp('2017-05-25 08:00'))&(ts<=pd.Timestamp('2017-05-30 23:59:59'));   # SHARED days (author 2026-09-30): the 6 h targets all ten stations share in the 15-30 May window, so no station is averaged over different days
        tsm=ts[mask]; ph=yp[mask]; hh=np.asarray(tsm.hour)
        q={}
        for qn,(h0,h1) in QH.items():
            mm=(hh>=h0)&(hh<h1) if h1!=24 else (hh>=h0); q[qn]=float(ph[mm].mean())
        qhat=conf[(conf.Station==sid)&(conf.Horizon=='6h')&(conf.Method=='Conformal')].Q_hat.iloc[0]
        rad=100/111320.0; circ=_Pt(sc['lon'],sc['lat']).buffer(rad); fp=bld.intersection(circ).area.sum()/circ.area*100
        ST[sid]=dict(lat=sc['lat'],lon=sc['lon'],lcz=sc['lcz'],lc=sc['lc'],elab=ELAB[sid],q=q,aft_off=q['Afternoon']-Kq['Afternoon'],qhat=float(qhat),fp=float(fp),bm=bm,skill=float(sk))
    pickle.dump((ST,Kq,LC_SKILL,bld,rg,lcz,meta,gtb),open(CACHE,'wb')); print('computed + cached')
print('ATL:',{k:round(v,1) for k,v in Kq.items()}); print('LC_SKILL:',{k:(v[0],round(v[1],2),round(v[2],1)) for k,v in LC_SKILL.items()})
CMEAN={q:float(np.mean([v['q'][q] for v in ST.values()])) for q in QH}   # per-quarter CAMPUS MEAN = the §4.6 deviation reference (pre-dawn 19.73 °C matches §4.6 exactly). REPLACES ATL for every per-site offset; ATL survives only as the '1 NWP cell' scale note in panel b. Keeps the figure intra-campus (no urban-rural UHI read).
print('CAMPUS MEAN:',{k:round(v,2) for k,v in CMEAN.items()})
QMAE={'Pre-dawn': 1.02, 'Morning': 1.92, 'Afternoon': 2.82, 'Evening': 1.18}   # 2026-09-30: shared days 25-30 May, canonical alignment (fig_script/_verify_quarter_mae.py); all-day 1.76   # RECOMPUTED 2026-08-26 by fig_script/_verify_quarter_mae.py (the previously cited _verify_afternoon_mae.py never existed anywhere in the project). Same code path as WMAE below, which it reproduces digit for digit on all ten stations; all-day mean = 1.7283 -> 1.73. Afternoon (2.5) = peak-solar hardest regime. Appended to the panel-a title (N4) so the shown quarter self-labels its accuracy.   [best-of-4: reverted from Option A per user]
WMAE={10635576: 1.77, 10635580: 2.06, 10635581: 1.57, 10635582: 1.83, 10635585: 1.75, 10635587: 1.77, 10635588: 1.71, 10635590: 1.61, 10635593: 1.82, 10635595: 1.74}   # 2026-09-30: per-station MAE on the shared days   # VERIFIED per-station 6h WINDOW MAE = Fig 9b column EXACTLY (best-of-4 per station; _verify_fig10_mae.py). Printed in each panel-c inset to tie each forecast to its validated accuracy.

lons=[v['lon'] for v in ST.values()]; lats=[v['lat'] for v in ST.values()]; gb=gtb.total_bounds; pad=0.0012
_mlon=max(lons)-min(lons); _mlat=max(lats)-min(lats); lo0=min(lons)-_mlon*0.15; lo1=max(lons)+_mlon*0.25; la0=min(lats)-120/111320.0; la1=max(lats)+140/111320.0   # bottom margin cut 200->120 m (user 2026-08-14 map-crop tightening; top keeps 200 m for site-3 label room)   # N-S edge = 100 m ring + 100 m offset = 200 m from the outermost stations
ASP=1.0/np.cos(np.radians((la0+la1)/2)); cmap=plt.cm.RdYlBu_r   # HARMONIZED with Fig 9 heatmaps (user 2026-08-14): same prediction, same diverging map, absolute 18-29 scale
from shapely.geometry import Polygon as _PolyB, MultiPolygon as _MPolyB
_cga = gtb.geometry.iloc[0] if len(gtb) == 1 else gtb.geometry.unary_union
_ppartsB = list(_cga.geoms) if _cga.geom_type == 'MultiPolygon' else [_cga]
_pamaxB = max(_pp.area for _pp in _ppartsB)
GTB_OUTER = _MPolyB([_PolyB(_pp.exterior) for _pp in _ppartsB if _pp.area >= 0.01 * _pamaxB])   # outer rings only, tiny parcels dropped (2026-09-05, as Fig 2)
QUARTER=sys.argv[1] if len(sys.argv)>1 else 'Afternoon'   # parameterise the deployment view by diurnal quarter
_NORMQ={'Pre-dawn':(18.0,29.0),'Morning':(18.0,29.0),'Afternoon':(18.0,29.0),'Evening':(18.0,29.0)}   # ONE absolute scale figure-wide (= floor/ceil of all quarter data = the strip axis)   # ALL strips 5 °C wide (consistent °C-per-pixel across quarters), each centred on its quarter's data + ATL band
_TICKQ={q2:list(range(18,30)) for q2 in _NORMQ}   # every-degree ticks 18..29, matching the panel-c strip axis
_TITLEQ={'Pre-dawn':'a) Campus coolest hours prediction T+6 h (Pre-dawn 00–06 h)','Morning':'a) Campus warm-up prediction T+6 h (Morning 06–12 h)','Afternoon':'a) Campus peak heating prediction T+6 h (Afternoon 12–18 h)','Evening':'a) Campus cool-down prediction T+6 h (Evening 18–24 h)'}   # APPLICATION-led titles (user 2026-08-16): what the model does, per quarter
norm=mcolors.Normalize(*_NORMQ[QUARTER])   # PER-QUARTER colour stretch (afternoon 24-29): map bubbles + inset chips + colorbar span the full colormap so the spatial hot/cool pattern reads, consistent with the strip's _NORMQ axis.
SHORT={10635593:'Tech Sq',10635585:'Baker',10635582:'Hemphill',10635587:'Curran',10635576:'10th St',10635580:'5th St',10635590:'Grass Wk',10635588:'Wetlands',10635581:'Canopied',10635595:'Brittain'}
LC_COLORS=dict(LC_COLORS); LC_COLORS['Open green']='#8CC63F'; LC_COLORS['Tree canopy']='#178B43'   # separate the two greens (light lime vs forest)

def draw_map(ax,qn,hero):
    # LCZ tint REMOVED — the LCZ classification now lives in Fig 2 panel d (a proper LCZ map). Panel a is a clean temperature map; LCZ stays as the strip stratifier (panel b) + per-inset label (panel c).
    ax.set_facecolor('#f5f5f0')
    bld.plot(ax=ax,facecolor='#e6e6e6',edgecolor='#c2c2c2',linewidth=0.12 if not hero else 0.25,alpha=0.9,zorder=2)   # buildings as light-grey FILLED footprints (solid blocks) so they're distinguishable from the line-drawn roads; still recede below the bubbles/rings
    rg.plot(ax=ax,color='#d8d8d8',linewidth=0.18 if not hero else 0.3,alpha=0.7,zorder=2)
    if hero:   # knock back the off-campus area so the campus reads as the figure (figure-ground; design review)
        import geopandas as _gpd; from shapely.geometry import box as _box
        _cg=GTB_OUTER
        _gpd.GeoSeries([_box(lo0,la0,lo1,la1).difference(_cg)]).plot(ax=ax,facecolor='white',edgecolor='none',alpha=0.55,zorder=3)
    _gpd.GeoSeries([GTB_OUTER.boundary]).plot(ax=ax,color='#8a8a8a',linewidth=0.75 if hero else 0.55,linestyle=(0,(6,3)),zorder=4)   # GT campus boundary RESTORED per user (the bottom clip where the campus exits south is normal/acceptable cartography)
    bs=300 if hero else 60   # hero bubbles enlarged so a one-decimal predicted-T value fits inside (was the site index)
    for sid,v in ST.items():
        lo,la=v['lon'],v['lat']
        _fc=cmap(norm(v['q'][qn]))
        if hero:   # panel-a markers = MINI SOURCE-AREA DISKS (match panel c): T-tinted 100 m disk with the campus buildings + roads showing through, dashed ring, centre sensor dot. Predicted-T value moves OUT to the leader (like panel c's footer) so the disk graphic stays clean.
            _rl=100/(111320*np.cos(np.radians(la))); _rt=100/111320.0
            _clip=Ellipse((lo,la),2*_rl,2*_rt,transform=ax.transData,facecolor='none',edgecolor='none',zorder=3.1); ax.add_patch(_clip)
            ax.add_patch(Ellipse((lo,la),2*_rl,2*_rt,facecolor=_fc,edgecolor='none',alpha=0.82,zorder=3.2))   # T tint fills the disk
            rg.plot(ax=ax,color='#9a9a9a',linewidth=0.5,alpha=0.9,zorder=3.3); ax.collections[-1].set_clip_path(_clip)   # roads inside the disk
            bld.plot(ax=ax,facecolor='#b3b3b3',edgecolor='#8f8f8f',linewidth=0.15,alpha=1.0,zorder=3.4); ax.collections[-1].set_clip_path(_clip)   # darker buildings inside the disk (contrast against the tint, like panel c)
            ax.add_patch(Ellipse((lo,la),2*_rl,2*_rt,fill=False,edgecolor='#8a8a8a',ls=(0,(2.2,2.2)),linewidth=0.8,zorder=4.5))   # dashed 100 m source-area ring
            ax.scatter(lo,la,s=26,facecolors='#1a1a2e',edgecolors='white',linewidths=0.8,zorder=6,path_effects=[pe.withStroke(linewidth=2.0,foreground='white')])   # centre sensor dot (panel-c style)
        else:
            ax.scatter(lo,la,s=bs,marker='o',c=[_fc],edgecolors='#333',linewidths=0.6,zorder=5,path_effects=[pe.withStroke(linewidth=1.8,foreground='white')])
            ax.text(lo,la,str(v['elab']),ha='center',va='center',fontsize=4.2,fontweight='bold',color='white',zorder=6,path_effects=[pe.withStroke(linewidth=0.9,foreground='black')])
        if hero:
            dx,dy=LABEL_OFFSETS[sid]
            _ab='%d · %s'%(v['elab'],v['lc']); _o=v['q'][qn]-CMEAN[qn]; _o=0.0 if abs(_o)<0.05 else _o   # signed-zero guard (no '-0.0') + index prefix preserves the a/b/c cross-ref. signed offset vs the CAMPUS MEAN (§4.6 reference; intra-campus, not the airport) — keeps the figure off the urban-rural UHI read.
            _os='%+.1f'%_o; _od=float(_os); _os=_os.replace('-','−')   # FIX (glyph collision): pick the ▲/▼/≈ glyph from the DISPLAYED rounded value, so two sites printing the same number can never carry opposite glyphs (was: raw _o vs ±0.5 disagreed with %.1f at threshold — site 4 ≈−0.5 vs site 10 ▼−0.5)
            _T='%.1f'%v['q'][qn]   # absolute predicted T now rides in the leader (moved out of the disk) — line 2 = 'T °C  ▲/▼/≈ Δ'
            if _od>0: _lab=_ab+chr(10)+_T+' °C  '+chr(9650)+' %s °C'%_os; _lc='#c0392b'   # BINARY sign convention (user 2026-08-13): ternary '≈' category removed — it grayed out Baker (+0.4) and the −0.5 vegetated sites, contradicting §4.6's 'runs warm'/'sit below' claims; honesty lives in ¶4's prose + printed values
            elif _od<0: _lab=_ab+chr(10)+_T+' °C  '+chr(9660)+' %s °C'%_os; _lc='#2166ac'
            else: _lab=_ab+chr(10)+_T+' °C 0.0 °C'; _lc='#4d4d4d'   # exact 0.0: UNSIGNED, single space, no glyph (audit 2026-08-17: '+0.0' put Tech Sq visually on the warm side)
            adx,ady=abs(dx),abs(dy); _D=28   # push labels clear of the 100 m circle
            if adx>=ady: _xt=(_D if dx>0 else -_D,0); _ha=('left' if dx>0 else 'right'); _va='center'
            else: _xt=(0,_D if dy>0 else -_D); _ha='center'; _va=('bottom' if dy>0 else 'top')
            ax.annotate(_lab,xy=(lo,la),xytext=_xt,textcoords='offset points',ha=_ha,va=_va,fontsize=FS_DENSE_SM,fontweight='bold',color=_lc,zorder=8,linespacing=1.25,path_effects=[pe.withStroke(linewidth=1.9,foreground='white')])
    ax.set_xlim(lo0,lo1); ax.set_ylim(la0,la1); ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect(ASP,adjustable='box')
    for s in ax.spines.values(): s.set_visible(False)   # panel-a map frame REMOVED (user 2026-08-18): frameless map, matching panel c's free-floating circles
    lab=_TITLEQ[qn]   # '· network MAE' suffix dropped (user 2026-09-11: plain panel labels; the 2.5 °C stays on panel d's tab, the legend box, and the caption)   # N4: 'network MAE' distinguishes this afternoon network mean (2.5) from the per-site MAEs in panel c (1.6–1.9) AND from panel b's 'spatial spread 2.5 °C' — the twin-2.5 disambiguation
    ax.set_title(lab,fontsize=FS_LABEL if hero else FS_DENSE_SM,fontweight='bold',loc='left',pad=2)
    if hero:
        # STAT BOX REMOVED (slimmed per user): the validation/skill/per-quarter-MAE were redundant with §4.6 + Figs 8/9 → moved to the CAPTION. Only the §4.6-binding consistency-check caveat stays on the hero map, as ONE 6 pt line (was sub-6pt in the box).
        pass  # panel-a consistency-check caveat REMOVED 2026-07-01 per user (declutter) -> baseline-reproducibility (r~0.92, n=40) goes in caption
        ax.annotate('N',xy=(0.965,0.955),xytext=(0.965,0.862),xycoords='axes fraction',ha='center',va='center',fontsize=8,fontweight='bold',zorder=9,arrowprops=dict(arrowstyle='-|>',lw=1.3,color='black'))   # N back to top-right corner (original) — stat box removed, corner is free again
        _sb=200/(111320*np.cos(np.radians((la0+la1)/2))); _x1=lo1-(lo1-lo0)*0.03; _x0=_x1-_sb; _yb=la0+(la1-la0)*0.07   # scale bar (200 m = one 100 m source-area ring diameter) in clear lower-right, subordinate to the rings
        ax.plot([_x0,_x1],[_yb,_yb],color='black',lw=1.4,zorder=9,solid_capstyle='butt',path_effects=[pe.withStroke(linewidth=2.8,foreground='white')]); ax.text((_x0+_x1)/2,_yb+0.0004,'200 m',ha='center',va='bottom',fontsize=FS_DENSE_SM,zorder=9,path_effects=[pe.withStroke(linewidth=2.2,foreground='white')])
        _ka=CMEAN[qn]   # legend swatch + reference colour = the CAMPUS MEAN (was ATL)
        from matplotlib.offsetbox import TextArea, DrawingArea, HPacker, VPacker, AnchoredOffsetbox
        from matplotlib.patches import Rectangle as _Rect, RegularPolygon as _RP
        from matplotlib.text import Text as _Txt
        _FS=FS_DENSE_SM
        def _lt(s,c='#1a1a2e',b=False,sz=None): return TextArea(s,textprops=dict(color=c,fontsize=(sz or _FS),fontweight=('bold' if b else 'normal')))
        def _gda(fn):
            d=DrawingArea(10,8,0,0); fn(d); return d
        def _row(g,lab): return HPacker(children=[g,_lt(lab)],align='center',pad=0,sep=2.5)
        # (orange campus-mean swatch _sw REMOVED — it was a phantom legend entry matching no glyph on the map; the campus-mean value is folded into the legend title below)
        _up=_gda(lambda d: d.add_artist(_RP((5,4),3,radius=3.0,orientation=0,facecolor='#c0392b',edgecolor='none')))
        _dn=_gda(lambda d: d.add_artist(_RP((5,4),3,radius=3.0,orientation=np.pi,facecolor='#2166ac',edgecolor='none')))
        _ap=_gda(lambda d: d.add_artist(_Txt(5,3.6,'≈',color='#555',fontsize=_FS+1,fontweight='bold',ha='center',va='center')))
        _ci=_gda(lambda d: d.add_artist(Circle((5,4),3.6,facecolor='none',edgecolor='#888',linewidth=0.7,linestyle=(0,(2,2)))))
        _key=VPacker(children=[_row(_up,'warmer than campus mean'),_row(_dn,'cooler than campus mean'),_row(_ci,'100 m deployment area')],align='left',pad=0,sep=1.2)   # phantom orange 'campus mean' square removed; value moved to the title line
        _offs=[v['q'][qn]-CMEAN[qn] for v in ST.values()]; _mno,_mxo=min(_offs),max(_offs)   # SIGNED two-sided range vs the CAMPUS MEAN — reads correctly in warm- or cool-dominant quarters
        _spn=_mxo-_mno; _mae=QMAE[qn]
        _rel=('spread %.1f °C = the %.1f °C network MAE'%(_spn,_mae)) if abs(_spn-_mae)<0.05 else ('spread %.1f °C vs network MAE %.1f °C'%(_spn,_mae))
        _take=_lt('Deviations vs the campus mean (%.1f °C)'%_ka+chr(10)+_rel+chr(10)+'context, not a model result','#1a1a2e',True)   # the frame travels WITH the claim (story review 2026-08-18)
        _box=VPacker(children=[_take,_key],align='left',pad=0,sep=3)
        _ab=AnchoredOffsetbox(loc='lower left',child=_box,bbox_to_anchor=(0.012,0.02),bbox_transform=ax.transAxes,frameon=True,pad=0.25,borderpad=0)
        _ab.patch.set(facecolor='white',edgecolor='#bbb',linewidth=0.5,alpha=0.95); _ab.patch.set_boxstyle('round,pad=0.25'); _ab.set_zorder(11)
        ax.add_artist(_ab)
        ax.text(0.998,0.006,'Basemap © OpenStreetMap contributors (ODbL)',transform=ax.transAxes,ha='right',va='bottom',fontsize=4.6,color='#b0b0b0',zorder=9,path_effects=[pe.withStroke(linewidth=1.5,foreground='white')])   # OSM/ODbL attribution required wherever the derived basemap (buildings+roads via osmnx) is shown; Elsevier production flags uncredited basemaps

def draw_inset(ax,sid,top):
    v=ST[sid]; lo,la=v['lon'],v['lat']; lc=v['lc']; cs=np.cos(np.radians(la))
    rlon=100/(111320*cs); rlat=100/111320.0          # TRUE 100 m circle (geographic) -> circular under ASP
    ex=108/(111320*cs); ey=108/111320.0              # extent half = 108 m -> 100 m circle ~fills the panel
    ax.set_xlim(lo-ex,lo+ex); ax.set_ylim(la-ey,la+ey)
    # predicted-T ring FILL restored per user — the "real temperature colour inside the circle" (cmap/norm; matches panel a + the colorbar). Buildings + sensor dot sit on top; the calibrated interval stays the text hero.
    ax.add_patch(Ellipse((lo,la),2*rlon,2*rlat,facecolor=cmap(norm(v['q'][QUARTER])),edgecolor='none',alpha=0.82,zorder=0))   # SAME opacity as panel a's disks (user 2026-08-15: fills must match exactly across panels)
    _clip2=Ellipse((lo,la),2*rlon,2*rlat,transform=ax.transData,facecolor='none',edgecolor='none',zorder=0.5); ax.add_patch(_clip2)   # clip roads+buildings to the 100 m circle (user 2026-08-15: nothing drawn outside the disk)
    rg.plot(ax=ax,color='#9a9a9a',linewidth=0.6,alpha=0.85,zorder=1); ax.collections[-1].set_clip_path(_clip2)
    bld.plot(ax=ax,facecolor='#b9b9b9',edgecolor='#8f8f8f',linewidth=0.25,alpha=0.95,zorder=2); ax.collections[-1].set_clip_path(_clip2)
    ax.set_xlim(lo-ex,lo+ex); ax.set_ylim(la-ey,la+ey)
    ax.add_patch(Ellipse((lo,la),2*rlon,2*rlat,fill=False,edgecolor='#888',ls='--',lw=0.9,zorder=4))
    ax.scatter(lo,la,s=9,facecolors='#1a1a2e',edgecolors='white',linewidths=0.7,zorder=5,path_effects=[pe.withStroke(linewidth=1.8,foreground='white')])   # SOLID sensor-pixel marker + white halo -> separates from dark building polygons on saturated (warm) insets
    # rotated side label REMOVED (user 2026-08-14): the 100 m source-area definition lives in panel a's legend + the caption
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect(ASP,adjustable='box')
    for s in ax.spines.values(): s.set_visible(False)   # square inset frame REMOVED (user 2026-08-15): free-floating clipped circles
    # #N badge REMOVED to declutter the source-area zoom (per user); the site index now lives in the title ('N · Station'). Cover-colour binding is carried by the panel-c column header above each inset.
    # Tier-1: predicted T as a SMALL corner chip (top-right) — colour still tied to the shared colorbar, but no longer the dominant ring fill
    # predicted-T value moved OFF the map into the subline (user: cleaner map); the ring FILL still carries the T colour
    # hierarchy: station = header hero; model+LCZ grey subline. All inset text now >= 6 pt (Elsevier floor); footprint% + skill% moved to the CAPTION to make room.
    ax.set_title('%d · %s'%(v['elab'],SHORT[sid]),fontsize=FS_DENSE_SM+0.5,fontweight='normal',color='#1a1a2e',loc='center',pad=5)   # station = REGULAR (entry); cover header = bold (group). temp moved to bottom line; pad 5 clears the cover rule   # index moved into the title ('N · Station') so the corner badge can go — keeps the a/b/c cross-ref
    # per-station MAE moved to the CAPTION (with model code + LCZ) — inset kept to cover+title / map / interval+vs-mean for breathing room   # N1: per-station best model code · LCZ · per-site MAE = Fig 9b column VERBATIM (2 dp) — each inset shows its OWN best model + validated accuracy
    _lo=v['q'][QUARTER]-v['qhat']; _hi=v['q'][QUARTER]+v['qhat']   # uniform 90% split-conformal PI = nowcast +/- per-station Q_hat
    _dtm=(lambda _x:0.0 if abs(_x)<0.05 else _x)(v['q'][QUARTER]-CMEAN[QUARTER]); _rdm=float('%+.1f'%_dtm)   # sign category from the DISPLAYED rounded value, same as panel a
    _dcm='#c0392b' if _rdm>0 else ('#2166ac' if _rdm<0 else '#666')   # BINARY sign convention (matches panel a; ternary '≈' removed per user 2026-08-13)
    from matplotlib.offsetbox import TextArea as _TA, HPacker as _HP, AnchoredOffsetbox as _AOB   # ONE line 'mean (90% PI) Δ' below the map — same read as panel b, colours preserved (mean dark / PI blue / Δ red-blue)
    def _seg(s,c,b=True): return _TA(s,textprops=dict(color=c,fontsize=FS_DENSE_SM,fontweight=('bold' if b else 'normal')))
    _line=_HP(children=[_seg('%.1f'%v['q'][QUARTER],'#1a1a2e'),_seg(' (%.0f–%.0f)'%(_lo,_hi),'#0072B2')],align='baseline',pad=0,sep=0)   # Delta segment DROPPED (2-step adoption): sign/family carried by the tree; exact deltas in panels a/b
    ax.add_artist(_AOB(loc='upper center',child=_line,bbox_to_anchor=(0.5,-0.05/_CIRC_K),bbox_transform=ax.transAxes,frameon=False,pad=0,borderpad=0))   # 2026-10-04: same gap under the smaller circle

def draw_qmap(ax,qn,lead=False):
    QHRS={'Pre-dawn':'00–06 h','Morning':'06–12 h','Afternoon':'12–18 h','Evening':'18–24 h'}
    CAB={'Tree canopy':'Tree','Open green':'Green','Built':'Built','Impervious':'Imperv','Exposed':'Exposed'}
    rgba=np.ones((*lcz.shape,4)); rgba[...,3]=0
    for vv,c in LCZC.items():
        mk=lcz==vv; r,g,b=mcolors.to_rgb(c); rgba[mk]=[r,g,b,0.10]
    ax.imshow(rgba,extent=[meta['lon_min'],meta['lon_max'],meta['lat_min'],meta['lat_max']],origin='upper',aspect='auto',zorder=0,interpolation='nearest')
    ax.set_facecolor('#f5f5f0')
    bld.plot(ax=ax,facecolor='#e8e8e8',edgecolor='none',alpha=0.7,zorder=2)
    import geopandas as _gpdB
    _gpdB.GeoSeries([GTB_OUTER.boundary]).plot(ax=ax,color='#3a3a4a',linewidth=0.7,zorder=4)
    for sid,v in ST.items():
        lo,la=v['lon'],v['lat']
        ax.scatter(lo,la,s=105,c=[cmap(norm(v['q'][qn]))],edgecolors='#333',linewidths=0.7,zorder=5)
        ax.text(lo,la,str(v['elab']),ha='center',va='center',fontsize=5,fontweight='bold',color='white',zorder=6,path_effects=[pe.withStroke(linewidth=1.0,foreground='black')])
        dx,dy=LABEL_OFFSETS[sid]
        ax.annotate(CAB[v['lc']],xy=(lo,la),xytext=(dx*0.95,dy*0.95),textcoords='offset points',ha='center',va='center',fontsize=4,fontweight='bold',color='#222',zorder=8,path_effects=[pe.withStroke(linewidth=1.2,foreground='white')],arrowprops=dict(arrowstyle='-',lw=0.3,color='#aaa',shrinkA=6,shrinkB=1))
    ax.set_xlim(lo0,lo1); ax.set_ylim(la0,la1); ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect(ASP,adjustable='box')
    for s in ax.spines.values(): s.set_edgecolor('#c0392b' if lead else '#bbb'); s.set_linewidth(2.0 if lead else 0.5)
    tm=np.mean([v['q'][qn] for v in ST.values()])
    ax.set_title('%s (%s) · mean %.1f °C'%(qn,QHRS[qn],tm)+('  — peak solar, hardest; deploy here' if lead else ''),fontsize=FS_DENSE_SM,fontweight='bold',loc='left',pad=2,color=('#c0392b' if lead else '#222'))

# DEAD 4-quarter overview block DISABLED 2026-07 (memory: rendered a full extra figure per run then discarded it). Delete on fig_script promotion.

def draw_scale_ladder(L):
    # OPTION B (forest-plot style): the 10 sites grouped TWO WAYS on one temperature axis so
    # "land cover explains more intra-campus variance than LCZ class" is SHOWN. Each row = mean
    # BOX + horizontal RANGE bar (member-site min-max; NOT a confidence interval) + site dots.
    #   by LAND COVER -> short bars (tight);  by LCZ class -> long bars (scattered).
    import numpy as _np
    from collections import defaultdict as _dd
    _ka=CMEAN[QUARTER]; _FS=FS_DENSE_SM
    _T0,_T1=_NORMQ[QUARTER]; _X0,_X1=0.230,0.790   # axis widened, number columns condensed (user 2026-08-16: more x-resolution -> less dot overlap)   # axis right edge pulled in to make room for the new 90% PI column
    def _tx(T): return _X0+(T-_T0)/(_T1-_T0)*(_X1-_X0)
    L.set_xlim(0,1); L.set_ylim(0,1); L.axis('off'); L.set_zorder(13); L.patch.set_visible(False)
    # panel-b frame box REMOVED (user 2026-08-14, height-reduction pass): no border; panels separate by whitespace alone
    _lcg=_dd(list); _lczg=_dd(list)
    for _v in ST.values(): _lcg[_v['lc']].append(_v); _lczg[str(_v['lcz'])].append(_v)
    def _mean(vs): return float(_np.mean([v['q'][QUARTER] for v in vs]))
    _lc_order=['Impervious', 'Open green', 'Built', 'Tree canopy', 'Exposed']   # re-pinned 2026-09-30 to the shared-days afternoon ranking   # PINNED (afternoon thermal ranking) for cross-quarter consistency
    _allT=[v['q'][QUARTER] for v in ST.values()]; _smin,_smax=min(_allT),max(_allT)
    def _delcol(dt): return '#c0392b' if dt>0 else ('#2166ac' if dt<0 else '#666')   # BINARY sign convention (matches panel a; ternary '≈' removed per user 2026-08-13)
    def _gq(vs): return float(_np.mean([v['qhat'] for v in vs]))   # mean 90% conformal half-width for a group (for the numeric PI column only)
    # --- title + land-cover legend ---
    _bb9=L.get_position(); fig.text(_x10[0],_bb9.y0+0.965*_bb9.height,'b) Two spatial scales',fontsize=FS_LABEL,fontweight='bold',color='#1a1a2e',ha='left',va='center',zorder=6)   # figure-anchored on the shared a/c/d left rail (review 2026-08-16)
    # land-cover legend REMOVED (user) — redundant: the named land-cover rows are themselves the colour key, reinforced by panel-c column headers
    # subtitle REMOVED (user) — forest plot SHOWS the finding; the eta2 claim + 'descriptive, n=10' hedge live in the caption/§4.6
    # --- campus-mean dashed reference line (the forest-plot 'null'/reference) + NEAR-MEAN BAND ---
    # near-mean band REMOVED (user 2026-08-13: grayed treatment blurred the results; binary sign convention adopted figure-wide)
    L.plot([_tx(_ka)]*2,[0.230,0.876],color='#444',lw=0.9,ls=(0,(3,2)),zorder=2.5)   # 2026-10-04: bottom follows the axis (was 0.190)
    L.text(_tx(_ka),0.910,'campus mean'+chr(10)+'%.1f °C'%_ka,fontsize=_FS,fontweight='bold',color='#444',ha='center',va='center',zorder=6,linespacing=1.12,path_effects=[pe.withStroke(linewidth=2.5,foreground='white')])
    def _row(y,vs,name,barcolor,flank=False,nums=True):   # nums=False: no site numerals (the LCZ rows, 2026-10-04)
        L.plot([_X0,_X1],[y,y],color='#f4f4f4',lw=0.5,zorder=1.5)
        _mv=_mean(vs); _dt=(0.0 if abs(_mv-_ka)<0.05 else round(_mv-_ka,1)); _ts=[v['q'][QUARTER] for v in vs]; _lo,_hi=min(_ts),max(_ts)   # delta from UNROUNDED values, rounded once for display (pub-review rounding convention: n=1 LCZ 2 row now matches its site's -0.8)
        _hw=_gq(vs); _pl,_ph=_mv-_hw,_mv+_hw   # per-group 90% conformal PI -> shown as the numeric column only (band removed: it dominated the temperature axis)
        if _hi>_lo:
            L.plot([_tx(_lo),_tx(_hi)],[y,y],color=barcolor,lw=2.6,zorder=3.5,solid_capstyle='round',path_effects=[pe.withStroke(linewidth=4.2,foreground='white')])
        _ds=sorted(vs,key=lambda v:v['q'][QUARTER]); _xs=[_tx(v['q'][QUARTER]) for v in _ds]
        _dyd=[0.0]*len(_xs)   # coincident-dot vertical dodge + dots ABOVE the mean square (review 2026-08-16: hidden site-10 dot, merged 24.4 pair)
        for _i9 in range(1,len(_xs)):
            if _xs[_i9]-_xs[_i9-1]<0.007: _dyd[_i9-1],_dyd[_i9]=0.0075,-0.0075
        for _v,_x,_d9 in zip(_ds,_xs,_dyd):
            L.scatter([_x],[y+_d9],s=13,facecolors=[LC_COLORS[_v['lc']]],edgecolors='white',linewidths=0.5,zorder=6.2,clip_on=False)
        _xm9=_tx(_ka)   # campus-mean line x: numerals must never sit on the dashes (audit 2026-08-17)
        if flank and len(_ds)==2:   # LC rows (n=2): numbers flank the pair — cooler-site left, warmer-site right, on the dot line (no vertical clutter)
            _xf0=_xs[0]-0.011; _xf1=_xs[-1]+0.011
            if abs(_xf0-_xm9)<0.013: _xf0=_xm9-0.013
            if abs(_xf1-_xm9)<0.013: _xf1=_xm9+0.013
            if _xf0!=_xs[0]-0.011: L.plot([_xs[0]-0.004,_xf0+0.002],[y,y],color='#cfcfcf',lw=0.4,zorder=3.9)   # leader when dash-dodged (consistency with LCZ rows, 2026-08-17)
            L.text(_xf0,y,str(_ds[0]['elab']),fontsize=_FS,color='#888',ha='right',va='center',zorder=6.3,path_effects=[pe.withStroke(linewidth=2.2,foreground='white')])
            if _xf1!=_xs[-1]+0.011: L.plot([_xs[-1]+0.004,_xf1-0.002],[y,y],color='#cfcfcf',lw=0.4,zorder=3.9)
            L.text(_xf1,y,str(_ds[-1]['elab']),fontsize=_FS,color='#888',ha='left',va='center',zorder=6.3,path_effects=[pe.withStroke(linewidth=2.2,foreground='white')])
        elif nums:   # LCZ rows (n=4,5): numbers above the dots with sideways dodge (off since 2026-10-04: the LCZ rows dropped their numerals)
            _lxn=list(_xs); _i=0
            while _i<len(_xs):
                _j=_i
                while _j+1<len(_xs) and _xs[_j+1]-_xs[_j]<0.030: _j+=1
                if _j>_i:
                    _c=sum(_xs[_i:_j+1])/(_j-_i+1)
                    for _z in range(_j-_i+1): _lxn[_i+_z]=_c+(_z-(_j-_i)/2.0)*0.030
                _i=_j+1
            for _r9 in range(3):   # relaxation: dash-avoid AND numeral-numeral separation (regression fix 2026-08-17: pre-dawn dodge overprinted 10 onto 4)
                for _i9 in range(len(_lxn)):
                    if abs(_lxn[_i9]-_xm9)<0.013: _lxn[_i9]+=(0.013 if _lxn[_i9]>=_xm9 else -0.013)
                for _i9 in range(1,len(_lxn)):
                    if _lxn[_i9]-_lxn[_i9-1]<0.011:
                        _mid9=(_lxn[_i9]+_lxn[_i9-1])/2.0
                        _lxn[_i9-1]=_mid9-0.0056; _lxn[_i9]=_mid9+0.0056
            for _v,_x,_xl in zip(_ds,_xs,_lxn):
                if abs(_xl-_x)>0.004: L.plot([_x,_xl],[y+0.010,y+0.021],color='#cfcfcf',lw=0.4,zorder=3.9,clip_on=False)
                L.text(_xl,y+0.023,str(_v['elab']),fontsize=_FS,color='#888',ha='center',va='bottom',zorder=6.3,path_effects=[pe.withStroke(linewidth=2.2,foreground='white')])   # halo + top zorder: dashes/mean-squares no longer strike the glyphs (review 2026-08-16)
        L.scatter([_tx(_mv)],[y],s=32,marker='s',facecolors='#1a1a2e',edgecolors='white',linewidths=0.7,zorder=6,clip_on=False)
        L.text(0.03,y,name,fontsize=_FS,color='#1a1a2e',va='center',ha='left',zorder=6)
        L.text(_X1+0.052,y,'%.1f'%_mv,fontsize=_FS,color='#555',va='center',ha='left',zorder=6)
        L.text(_X1+0.088,y,'(%.0f–%.0f)'%(_pl,_ph),fontsize=_FS,color='#0072B2',va='center',ha='left',zorder=6)   # 90% PI in parens, conformal blue -> one combined 'mean (90% PI)' cell
        L.text(_X1+0.146,y,('0.0' if _dt==0 else ('%+.1f'%_dt).replace('-','−')),fontsize=_FS,fontweight='bold',color=_delcol(_dt),va='center',ha='left',zorder=6)   # unsigned zero (audit 2026-08-17)
        L.text(_X1+0.013,y,('%.1f'%(_hi-_lo)) if len(vs)>1 else '–',fontsize=_FS,color='#555',va='center',ha='left',zorder=6)   # RANGE column (user 2026-08-16): quantifies tight-covers-vs-scattered-classes; puts §4.6's 2.2 within-LCZ-5 on the figure
    # --- block 1: by land cover (tight) — with forest-plot key on the header row ---
    _cms=[_mean(_lcg[_k]) for _k in _lc_order]; _lzs=[_mean(_lczg[_z]) for _z in ['2','5','8']]
    L.text(0.03,0.905,'grouped by land cover (means span %.1f °C)'%(max(_cms)-min(_cms)),fontsize=_FS,fontweight='bold',color='#1a1a2e',va='center',ha='left',zorder=6)
    # block rules REMOVED (user 2026-08-13): bold group labels alone separate the two blocks — cleaner minimal look
    L.scatter([0.030],[0.206],s=20,marker='s',facecolors='#1a1a2e',edgecolors='white',linewidths=0.6,zorder=6,clip_on=False); L.text(0.045,0.206,'mean',fontsize=_FS,color='#888',va='center',ha='left',zorder=6)   # key moved beside the axis row; spread arrow REMOVED (user 2026-08-16: duplicated panel d's emphasized bar)
    L.plot([0.100,0.135],[0.206,0.206],color='#9aa0aa',lw=2.6,zorder=6,clip_on=False,solid_capstyle='round'); L.text(0.145,0.206,'site range',fontsize=_FS,color='#888',va='center',ha='left',zorder=6)
    L.text(_X1+0.052,0.905,'mean',fontsize=_FS,color='#999',va='center',ha='left',zorder=6); L.text(_X1+0.088,0.905,'(90% PI)',fontsize=_FS,color='#0072B2',va='center',ha='left',zorder=6); L.text(_X1+0.146,0.905,'Δ campus',fontsize=_FS,color='#999',va='center',ha='left',zorder=6); L.text(_X1+0.013,0.905,'range',fontsize=_FS,color='#999',va='center',ha='left',zorder=6)
    _yL=[0.838,0.766,0.694,0.622,0.550]
    for _y,_lc in zip(_yL,_lc_order): _row(_y,_lcg[_lc],_lc,LC_COLORS[_lc],flank=True)
    # --- block 2: by LCZ class (scattered) ---
    L.text(0.03,0.486,'grouped by LCZ class (means span %.1f °C)'%(max(_lzs)-min(_lzs)),fontsize=_FS,fontweight='bold',color='#1a1a2e',va='center',ha='left',zorder=6)
    _MORPH={'2':'compact mid-rise','5':'open mid-rise','8':'large low-rise'}
    _yZ=[0.424,0.352,0.280]   # 2026-10-04: numerals gone -> back to the land-cover row spacing (0.072); were [0.424,0.332,0.240] (opened up 2026-08-16 for the numeral rows)
    for _y,_z in zip(_yZ,['2','5','8']):
        _row(_y,_lczg[_z],'LCZ %s'%_z,'#9aa0aa',nums=False)   # 2026-10-04 (author): no site numerals on the LCZ rows; cover colour + the land-cover rows' end numerals identify the sites
        L.text(0.078,_y,'(%s)  n = %d'%(_MORPH[_z],len(_lczg[_z])),fontsize=_FS,color='#9a9a9a',va='center',ha='left',zorder=6)
    # skill strapline REMOVED (user 2026-09-11): it was a result claim with a stale §4.2 pointer drawn on the figure face and its descenders collided with the panel-c title; skill vs the benchmarks is cross-referenced in the caption (§4.5, §4.6), per the 2026-07-01 caption rule   # the benchmark a benchmark paper's capstone must show; QUALITATIVE on purpose - LC_SKILL aggregates on a different basis than §4.6's 7-13% (cover means), so printing a number here would put a THIRD variant on the page (story review 2026-08-18)
    # --- shared temperature axis ---
    _ax=0.230   # 2026-10-04: axis follows the closer LCZ rows (was 0.190)
    L.plot([_X0,_X1],[_ax,_ax],color='#aaa',lw=0.6,zorder=3)
    for _tk in _TICKQ[QUARTER]:
        L.plot([_tx(_tk)]*2,[_ax,_ax-0.014],color='#aaa',lw=0.6,zorder=3); L.text(_tx(_tk),_ax-0.024,'%d'%_tk,fontsize=_FS,color='#777',ha='center',va='top',zorder=3)
    L.text(_X1+0.014,_ax-0.024,'°C',fontsize=_FS,color='#777',ha='left',va='top',zorder=3)

fig=plt.figure(figsize=(FULL_W,9.4))
_CIRC_K=0.70   # panel-c map circles at 70% of their 2026-08 size (author 2026-10-04: 'reduce the map circle 30% to reduce the fig height')
_DYC=(1-_CIRC_K)*(0.2129-0.1429)   # height the circles give up (figure fraction): everything ABOVE them moves down by it; the tight crop takes it off the top
_DYB=0.040*0.225   # panel-b LCZ rows closed up by 0.040 of the box (2026-10-04, numerals dropped): everything above them moves down by this too
_DYL=0.2499-0.2394   # panel-c LCZ tag row removed (2026-10-04, author: 'LCZ is not necessary in this plot'): rails and everything above them move down by it
_RAIL_K=0.65   # cover-connector rail set at 65% of its 2026-08 height (author 2026-10-04: 'reduce this line set 35% in total')
_DYR=(1-_RAIL_K)*((0.2615-0.2499)+0.00945*3)   # height saved for the most crowded quarter (4 rail levels, Afternoon/Evening): everything above moves down by it
_hfrac=(0.91*FULL_W)*((la1-la0)*ASP/(lo1-lo0))/9.4   # exact drawn-map height -> box hugs the map, savings bubble to the TOP edge (bbox-cropped)
axh=fig.add_axes([0.015,0.523-_DYC-_DYB-_DYL-_DYR,0.91,_hfrac]); draw_map(axh,QUARTER,True); axh.set_anchor('NW')   # panel a shifted down 0.041 with panel b's shrink -> whitespace moves to the TOP edge where tight-bbox crops it; a-b seam unchanged   # right edge pulled in to 0.925 so the vertical colorbar (x=0.945) gets its own clear column — no overlap with panel a
# --- panel c: TEMPERATURE-SCALE ARCS (ADOPTED 2026-08-13, user decision; supersedes the 2-step tree): maps sorted cool->warm, per-leaf cover name + LCZ tag, same-cover sites joined by arcs (a split cover\u2019s arc crosses the campus-mean divider = the dual-scale punchline), magnified per-quarter bar RETIRED 2026-08-15 (redundant with the strip). ---
_cmz=CMEAN[QUARTER]
_order=sorted(ZOOM,key=lambda s:ST[s]['q'][QUARTER])
_nCool=sum(1 for s in _order if ST[s]['q'][QUARTER]<_cmz)
_w10=0.97/10.9; _s10n=0.10*_w10-0.018/9.0; _GDIV=0.018   # COOL/WARM FAMILY GAP (user 2026-08-15): map width unchanged, inter-map spacing tightened to fund an extra 0.018 at the divider (~3.6x normal gap) - proximity separates the families, the split-cover rail visibly leaps it
_x10=[0.015+_k*(_w10+_s10n)+(_GDIV if _k>=_nCool else 0.0) for _k in range(10)]
_gy0,_gy1=0.1429,0.1429+_CIRC_K*(0.2129-0.1429)   # circle bottoms stay put (2026-10-04); tops come down
_LEFTCOL={_order[0]}
_colax=[]
for _ci,_s in enumerate(_order):
    _a=fig.add_axes([_x10[_ci]+(1-_CIRC_K)*_w10/2,_gy0,_CIRC_K*_w10,_gy1-_gy0]);   # 2026-10-04: smaller circle, same slot centre
    draw_inset(_a,_s,True); _colax.append(_a)
fig.canvas.draw()
hpos=axh.get_position()
cax=fig.add_axes([0.945,hpos.y0,0.012,hpos.height]); sm=plt.cm.ScalarMappable(cmap=cmap,norm=norm); sm.set_array([])
cb=fig.colorbar(sm,cax=cax,orientation='vertical'); cb.set_label('Predicted T+6 h (°C)',fontsize=FS_DENSE_SM,labelpad=2); cb.set_ticks(_TICKQ[QUARTER]); cb.ax.tick_params(labelsize=FS_TICK); cb.outline.set_linewidth(0.6)
_lax=fig.add_axes([0.015,0.288-_DYC-_DYB-_DYL-_DYR,0.97,0.225]); draw_scale_ladder(_lax)   # height-reduction lever 1 (user 2026-08-14): panel-b box shrunk 0.266->0.225 (rows compress ~15%, text stays at point size)
_cc=[_x10[_k]+_w10/2 for _k in range(10)]
_hy=_gy1
_DKLC=dict(LC_COLORS); _DKLC['Open green']='#5a9e1f'   # darker lime for text legibility
for _ci,_s in enumerate(_order):
    _lc=ST[_s]['lc']
    fig.text(_cc[_ci],0.2289-_DYC,_lc,fontsize=FS_DENSE_SM,fontweight='bold',color=_DKLC[_lc],ha='center',va='bottom',path_effects=[pe.withStroke(linewidth=2.0,foreground='white')],zorder=7)

# cover-connection arcs (arc-diagram idiom): each cover's two maps joined in its colour;
# Built's arc CROSSES the divider = the broken pair leaping the family boundary

# per-station 'LCZ x' tags REMOVED 2026-10-04 (author: 'LCZ is not necessary in this plot'): LCZ stays in panel b's grouped rows;
# each station's class is in Fig. 2c and Table S14

# cover connectors: CIRCUIT-DIAGRAM RAILS (user 2026-08-13; replaces bezier arcs): straight
# rails at staggered heights (greedy interval coloring, per quarter); wire-jump gaps where a
# higher rail's vertical passes a lower rail; the split cover's rail crosses the divider.
_y0c=0.2499-_DYC-_DYL   # 2026-10-04: rails start where the LCZ tag row was
_slot={s2:i2 for i2,s2 in enumerate(_order)}
_covpairs={}
for _s in ZOOM: _covpairs.setdefault(ST[_s]['lc'],[]).append(_s)
_ends={lc:tuple(sorted((_cc[_slot[ss[0]]],_cc[_slot[ss[1]]]))) for lc,ss in _covpairs.items()}
_lvl={}
for _lc in sorted(_ends,key=lambda k:_ends[k][1]-_ends[k][0]):
    _x1,_x2=_ends[_lc]; _L=0
    while any(_lvl[_o]==_L and not(_ends[_o][1]<=_x1 or _ends[_o][0]>=_x2) for _o in _lvl): _L+=1
    _lvl[_lc]=_L
_LVLH={lc:_y0c+_RAIL_K*((0.2615-0.2499)+0.00945*_lvl[lc]) for lc in _lvl}   # 2026-10-04: leg height and level spacing both x _RAIL_K (rail zone compressed 30% on 2026-08-14)   # rail zone compressed 30% (user 2026-08-14)
def _vln2(x,y0,y1,c): fig.add_artist(plt.Line2D([x,x],[y0,y1],color=c,lw=1.5,zorder=5.5,solid_capstyle='butt'))
for _lc,(_x1,_x2) in _ends.items():
    _h=_LVLH[_lc]; _c2=_DKLC[_lc]
    _vln2(_x1,_y0c,_h,_c2); _vln2(_x2,_y0c,_h,_c2)
    _cuts=sorted(set(_x for _olc,(_xa,_xb) in _ends.items() if _LVLH[_olc]>_h for _x in (_xa,_xb) if _x1<_x<_x2))
    _segs=[_x1]+[v for _x in _cuts for v in (_x-0.005,_x+0.005)]+[_x2]
    for _i2 in range(0,len(_segs),2):
        fig.add_artist(plt.Line2D([_segs[_i2],_segs[_i2+1]],[_h,_h],color=_c2,lw=1.5,zorder=5.5,solid_capstyle='butt'))
_xdv=(_x10[_nCool-1]+_w10+_x10[_nCool])/2   # centre of the widened family gap
fig.add_artist(plt.Line2D([_xdv,_xdv],[0.2264-_DYC,0.1219],color='#444',lw=1.0,ls=(0,(3,2)),zorder=4.6,alpha=0.9))   # top ends LEVEL with the LCZ tag row (user 2026-08-16, cleaner; dash phase starts at top so the terminus is a full dash)
fig.add_artist(plt.Line2D([_xdv-0.004,_xdv+0.004],[0.2264-_DYC,0.2264-_DYC],color='#444',lw=1.0,zorder=4.6,alpha=0.9,solid_capstyle='butt'))   # head cap at the divider terminus (user 2026-08-16)
# FAMILY ROW (user 2026-08-16): two-line value/label triplet - cool-group mean, campus mean, warm-group mean - with block-spanning arms; the three values align on one line = 'mean + resolved structure' in miniature
_cmC=float(np.mean([ST[_s9]['q'][QUARTER] for _s9 in _order[:_nCool]]))
_cmW=float(np.mean([ST[_s9]['q'][QUARTER] for _s9 in _order[_nCool:]]))
_yfam=0.1119
for _fx0,_fx1,_fv,_fw,_fcol in [(_x10[0]+0.006,_xdv-0.036,_cmC,'cooler','#2166ac'),(_xdv+0.036,_x10[9]+_w10-0.006,_cmW,'warmer','#c0392b')]:
    fig.add_artist(plt.Line2D([_fx0,_fx1],[_yfam,_yfam],color=_fcol,lw=0.9,zorder=7,solid_capstyle='butt'))
    _fxo=_fx0 if _fv==_cmC else _fx1
    fig.add_artist(plt.Line2D([_fxo,_fxo],[_yfam,_yfam+0.004],color=_fcol,lw=0.9,zorder=7,solid_capstyle='butt'))
    fig.text((_fx0+_fx1)/2,_yfam,'%.1f °C'%_fv+chr(10)+_fw,fontsize=FS_DENSE_SM,fontweight='bold',color=_fcol,ha='center',va='center',zorder=8,linespacing=1.15,bbox=dict(facecolor='white',edgecolor='none',pad=1.4))
fig.text(_xdv,_yfam,'%.1f °C'%_cmz+chr(10)+'campus mean',fontsize=FS_DENSE_SM,fontweight='bold',color='#444',ha='center',va='center',zorder=8,linespacing=1.15,bbox=dict(facecolor='white',edgecolor='none',pad=1.4))

# magnified per-quarter bar RETIRED (user 2026-08-15): redundant once the diurnal strip gained the identical bar design + per-site ticks; exact values live in the footers, the metric scale in the strip.
# --- DIURNAL CONTEXT STRIP (user 2026-08-14): all four quarters' campus min-max + mean on one
# common ABSOLUTE temperature scale; the displayed quarter is emphasized and zoom-linked to the
# magnified detail bar above -> cross-quarter comparison at a glance. ---
_qstats={qn:(min(ST[_s3]['q'][qn] for _s3 in ZOOM),max(ST[_s3]['q'][qn] for _s3 in ZOOM),float(np.mean([ST[_s3]['q'][qn] for _s3 in ZOOM]))) for qn in QH}
_S0,_S1=_NORMQ[QUARTER]   # PINNED to the figure-wide absolute bounds (user 2026-08-14: 18-29; one scale everywhere)
def _sx(T): return 0.045+(T-_S0)/(_S1-_S0)*0.91
_normG=mcolors.Normalize(_S0,_S1)
_qlv={}
for _qn in sorted(QH,key=lambda k:_qstats[k][0]):
    _a,_b=_qstats[_qn][0],_qstats[_qn][1]; _L=0
    while any(_qlv[_o]==_L and not(_qstats[_o][1]<=_a or _qstats[_o][0]>=_b) for _o in _qlv): _L+=1
    _qlv[_qn]=_L
_QY={0:0.0336,1:-0.0475}; _QBH=0.0065   # Evening raised: mirror layout needs no tick-clearance below (near-equal axis gaps); _QBH=0.0065   # row swap trialed then REVERTED (user 2026-08-15): trio on TOP, Evening below; doubled gap kept; _QBH=0.0065   # level-1 row lowered so its label fits top-left (user 2026-08-15); _QBH=0.0065   # strip shifted down with the relocated mean label (user 2026-08-14)   # level-0 raised: inter-level gap ~4.4pt (user 2026-08-14), axis gap below unchanged   # strip shifted down 6pt then 7pt more (user 2026-08-14/15): clear gap from the detail bar marks it as a separate diurnal-info layer
fig.add_artist(plt.Line2D([0.045,0.955],[-0.0018,-0.0018],color='#aaa',lw=0.6,zorder=3))   # axis moved ABOVE the bars (user 2026-08-15 swap)
_lastsx=None
for _tk in np.arange(np.ceil(_S0),_S1+1e-9,1.0):
    fig.add_artist(plt.Line2D([_sx(_tk)]*2,[-0.0058,-0.0018],color='#aaa',lw=0.5,zorder=3))
    fig.text(_sx(_tk),-0.0068,'%d'%_tk,fontsize=FS_DENSE_SM,color='#777',ha='center',va='top')   # numbers BELOW the line, standard (user 2026-08-15)
    _lastsx=_sx(_tk)
fig.text(_lastsx+0.014,-0.0068,'°C',fontsize=FS_DENSE_SM,color='#777',ha='left',va='top')
_QHRSS={'Pre-dawn':'00–06 h','Morning':'06–12 h','Afternoon':'12–18 h','Evening':'18–24 h'}
_QAPP={'Pre-dawn':'Coolest hours (Pre-dawn)','Morning':'Warm-up (Morning)','Afternoon':'Peak heating (Afternoon)','Evening':'Cool-down (Evening)'}   # application name first, quarter name as the bridge to the text/caption (user 2026-09-11); spelling matches the panel-a titles   # same strings as panel a's header
for _qn in QH:
    _a,_b,_m=_qstats[_qn]; _y=_QY[_qlv[_qn]]; _cur=(_qn==QUARTER); _bel=_y<0.0019   # below-axis bar -> mirrored layout
    _xq=np.linspace(_sx(_a),_sx(_b),41)
    for _i3 in range(40):
        fig.add_artist(plt.Rectangle((_xq[_i3],_y),_xq[_i3+1]-_xq[_i3]+0.0004,_QBH,facecolor=cmap(_normG(_a+(_b-_a)*(_i3+0.5)/40.0)),edgecolor='none',zorder=4))
    fig.add_artist(plt.Rectangle((_sx(_a),_y),_sx(_b)-_sx(_a),_QBH,fill=False,edgecolor=('#444' if _cur else '#999'),lw=(0.9 if _cur else 0.4),zorder=4.5))
    for _s4 in ZOOM:   # per-site tick marks (user 2026-08-14): every quarter shows its full site distribution; the magnified bar above carries the numbered version
        fig.add_artist(plt.Line2D([_sx(ST[_s4]['q'][_qn])]*2,[_y,_y+_QBH],color='#333',lw=0.45,alpha=0.75,zorder=4.8))   # tick halo trialed then REVERTED (user 2026-08-15: keep plain ticks)
    fig.add_artist(plt.Line2D([_sx(_m)]*2,[_y-(0.0058 if _bel else 0.0015),_y+_QBH+(0.0015 if _bel else 0.0058)],color='#1a1a2e',lw=1.1,zorder=5))   # tick extends toward the offset mean number; mirrored side shorter - digits have descent space below but none above cap height (visual-gap symmetry, 2026-08-17)   # solid mean tick (dashed variant trialed then REVERTED, user 2026-08-15)
    for _xe2 in (_sx(_a),_sx(_b)):   # dotted end connectors bar->axis, panel-b spread-arrow idiom (user 2026-08-15)
        fig.add_artist(plt.Line2D([_xe2,_xe2],([-0.0018,_y] if _y>-0.0018 else [_y+_QBH,-0.0018]),color='#9aa0aa',lw=0.7,ls=(0,(1,1.5)),zorder=3.2))   # axis->nearest bar edge (Evening now BELOW the axis, user 2026-08-15)   # connectors rise to the top axis; occluded where they pass behind a bar
    fig.text(_sx(_m),(_y-0.0077 if _bel else _y+_QBH+0.0077),'%.1f °C'%_m,fontsize=FS_DENSE_SM,fontweight=('bold' if _cur else 'normal'),color=('#333' if _cur else '#888'),ha='center',va=('top' if _bel else 'bottom'),zorder=6,bbox=dict(boxstyle='round,pad=0.30',facecolor='white',edgecolor=('#555' if _cur else '#c8c8c8'),linewidth=(1.0 if _cur else 0.5)))   # mean value in a white fillet-corner chip, offset +1.5pt (user 2026-08-17)
    _ya2=(((_y+_QBH)-0.0018)/2.0-0.00115 if _bel else (_y-0.0018)/2.0+0.00115)   # BOTH rows rigidly 1.5pt further from the axis (user 2026-08-18, mirrored): the extra term keeps each tab moving WITH its bar, so internal spacing is unchanged on both sides   # tab CENTERED between bar edge and the 18-29 axis (user 2026-08-17)   # dimension arrow on the AXIS side for mirrored bars
    _thh=0.0058   # FLAT-ended ribbon (user 2026-08-18: triangular arrow caps trialed then REVERTED); thick enough to enclose the 6 pt text; grey darkened 25%
    _rx0,_rx1=_sx(_a),_sx(_b); _RMIN=0.178
    if _rx1-_rx0<_RMIN: _rc=(_rx0+_rx1)/2.0; _rx0,_rx1=_rc-_RMIN/2.0,_rc+_RMIN/2.0   # label never overflows its ribbon (2026-09-30)
    fig.add_artist(plt.Rectangle((_rx0,_ya2-_thh),_rx1-_rx0,2*_thh,facecolor='#686e75',edgecolor='none',zorder=5.5))
    fig.text((_sx(_a)+_sx(_b))/2,_ya2,'spread %.1f °C | MAE %.1f °C'%(_b-_a,QMAE[_qn]),fontsize=FS_DENSE_SM,fontweight='bold',color='white',ha='center',va='center',zorder=6)
    _dg7=sorted(((_sx(ST[_s4]['q'][_qn]),LC_COLORS[ST[_s4]['lc']]) for _s4 in ZOOM))   # LC dots in the bar-to-tab gap (user 2026-08-17): cover identity per site, on white for legibility
    _dxs=[_d8[0] for _d8 in _dg7]
    for _i8 in range(1,len(_dxs)):
        if _dxs[_i8]-_dxs[_i8-1]<0.0080: _dxs[_i8]=_dxs[_i8-1]+0.0080   # micro-dodge exact ties
    _ydot=(((_y+_QBH)+(_ya2-_thh))/2.0 if _bel else (_y+(_ya2+_thh))/2.0)   # dots centered in the widened bar-to-tab band
    for _dx8,_dg8 in zip(_dxs,_dg7):
        _bed8=(_y+_QBH if _bel else _y); _ded8=(_ydot-0.0028 if _bel else _ydot+0.0028)
        fig.add_artist(plt.Line2D([_dg8[0],_dx8],[_bed8,_ded8],color='#bbb',lw=0.4,zorder=5.3))   # hair stem: dot to its true position on the bar (user 2026-08-17)
        fig.add_artist(plt.Line2D([_dx8],[_ydot],marker='o',ms=4.1,mfc=_dg8[1],mec='white',mew=0.5,ls='None',zorder=5.6))
    fig.text(_sx(_m),(_y-0.0307 if _bel else _y+_QBH+0.0290),_QAPP[_qn],fontsize=FS_DENSE_SM,fontweight=('bold' if _cur else 'normal'),color=('#333' if _cur else '#888'),ha='center',va=('top' if _bel else 'bottom'),zorder=6)   # quarter NAME atop the node stack (user 2026-08-18)
    fig.text(_sx(_m),(_y-0.0215 if _bel else _y+_QBH+0.0198),_QHRSS[_qn],fontsize=FS_DENSE_SM,fontweight=('bold' if _cur else 'normal'),color=('#333' if _cur else '#888'),ha='center',va=('top' if _bel else 'bottom'),zorder=6)   # hours above the mean chip, outside the box (user 2026-08-17)   # top-left labels + hour ranges (panel-a convention); halo dropped + offset raised so descenders never notch the bar outline (review 2026-08-15)
fig.text(_x10[0],0.296-_DYC-_DYL-_DYR,'c) Cooler and warmer micro-environments',ha='left',va='bottom',fontsize=FS_LABEL,fontweight='bold',color='#1a1a2e')   # 'on the temperature scale' dropped when the strip became panel d (user 2026-08-15)
_TRAV=max(CMEAN.values())-min(CMEAN.values()); _MAXSP=max(max(ST[_s]['q'][_q] for _s in ZOOM)-min(ST[_s]['q'][_q] for _s in ZOOM) for _q in QH)
fig.text(_x10[0],0.0831,'d) T+6 h diurnal temperature and across-site spread',ha='left',va='bottom',fontsize=FS_LABEL,fontweight='bold',color='#1a1a2e')
# T+6h CHAIN (user 2026-08-17): connectors for ALL consecutive quarter pairs, start cap '|' +
# arrowhead, each labeled with its hop cost. Same-row pairs run text-to-text at the number row;
# Afternoon->Evening takes the cross-axis elbow. Pre-dawn's incoming hop (previous day's Evening)
# is stated as a note in that render only.
_ARR=dict(color='#555',lw=0.8,ls=(0,(2,1.5)),zorder=5.8,solid_capstyle='butt')   # chips and hop labels carry °C (user 2026-09-11); cap/arrow offsets widened 0.028/0.030/0.032 -> 0.036/0.038/0.040 to keep the chip clearance
def _bl6(x0,y0,x1,y1): fig.add_artist(plt.Line2D([x0,x1],[y0,y1],**_ARR))
_yTOP=_QY[0]; _yEVE=_QY[1]
for _qa7,_qb7 in [('Pre-dawn','Morning'),('Morning','Afternoon'),('Afternoon','Evening'),('Evening','Pre-dawn')]:
    _d7=round(float('%.1f'%CMEAN[_qb7])-float('%.1f'%CMEAN[_qa7]),1); _xa7=_sx(CMEAN[_qa7]); _xb7=_sx(CMEAN[_qb7]); _sfx7=''   # hop = difference of the DISPLAYED chips (review 2026-09-11: readers subtract the chips; unrounded gave +3.4/+4.4 vs chips 3.5/4.3)
    if _qb7!='Evening' and _qa7!='Evening':
        _yl7=_yTOP+_QBH+0.0114
        fig.add_artist(plt.Line2D([_xa7+0.036]*2,[_yl7-0.0018,_yl7+0.0018],color='#555',lw=0.8,zorder=5.9,solid_capstyle='butt'))
        _bl6(_xa7+0.036,_yl7,_xb7-0.038,_yl7)
        fig.add_artist(plt.Line2D([_xb7-0.036],[_yl7],marker='>',ms=2.6,color='#555',mew=0,ls='None',zorder=5.9))
        _lx7,_ly7=(_xa7+_xb7)/2.0,_yl7
    elif _qa7=='Evening':   # 4th hop (user 2026-08-17): Evening -> next day's Pre-dawn, around the LEFT side - closes the diurnal circuit across midnight
        # SPLIT (user 2026-09-11): the closed loop sent Evening back into the SAME day's Pre-dawn bar; the hop
        # really lands in the NEXT day's Pre-dawn. Now an outgoing stub at Evening (labeled) + an incoming stub at Pre-dawn.
        _yl7=_yTOP+_QBH+0.0114; _yE7=_yEVE-0.0114; _x0e=_xa7-0.040; _sfx7=') to next day' + chr(8217) + 's Pre-dawn'
        _xend=0.075   # outgoing stub ends at the same x where the incoming stub starts (user 2026-09-11: line the two up on the left)
        fig.add_artist(plt.Line2D([_x0e]*2,[_yE7-0.0018,_yE7+0.0018],color='#555',lw=0.8,zorder=5.9,solid_capstyle='butt'))
        _bl6(_x0e,_yE7,_xend,_yE7)
        _bl6(_xend,_yE7,_xend,_yl7); _bl6(_xend,_yl7,_xb7-0.038,_yl7)   # plain corner up the left margin and into the Pre-dawn chip (panel-d review 2026-09-11: no node, no 'input' - the chips are means of DIRECT forecasts, nothing feeds anything)
        fig.add_artist(plt.Line2D([_xb7-0.036],[_yl7],marker='>',ms=2.6,color='#555',mew=0,ls='None',zorder=5.9))   # arrowhead at the Pre-dawn chip, like the other three hops
        _lx7,_ly7=(_x0e+_xend)/2.0,_yE7
    else:   # Afternoon -> Evening, TEXT-TO-TEXT (user 2026-08-17): from the end of the 26.7 text,
        # around the right side, arriving at the end of the 23.0 text below the mirrored Evening bar
        _yl7=_yTOP+_QBH+0.0114
        _x0e=_xa7+0.038
        _xsA=sorted(_sx(ST[_s9]['q']['Afternoon']) for _s9 in ZOOM)   # same 0.0080 micro-dodge as the dot row
        for _i9 in range(1,len(_xsA)):
            if _xsA[_i9]-_xsA[_i9-1]<0.0080: _xsA[_i9]=_xsA[_i9-1]+0.0080
        _xr7=max(_sx(_qstats['Afternoon'][1]),_xsA[-1])+0.012   # down-leg placed from the data: clear of the Afternoon bar end AND its dodged site dots (user 2026-09-11: the fixed 0.895 ran through the bar)
        for _tv7 in np.arange(np.ceil(_S0),_S1+1e-9,1.0):
            if abs(_sx(_tv7)-_xr7)<0.013: _xr7=_sx(_tv7)+0.016; break   # down-leg dodges axis numbers
        _yE7=_yEVE-0.0077-0.0037   # Evening mean-text centre
        fig.add_artist(plt.Line2D([_x0e]*2,[_yl7-0.0018,_yl7+0.0018],color='#555',lw=0.8,zorder=5.9,solid_capstyle='butt'))
        _bl6(_x0e,_yl7,_xr7,_yl7); _bl6(_xr7,_yl7,_xr7,_yE7); _bl6(_xr7,_yE7,_xb7+0.040,_yE7)
        fig.add_artist(plt.Line2D([_xb7+0.038],[_yE7],marker='<',ms=2.6,color='#555',mew=0,ls='None',zorder=5.9))
        _lx7,_ly7=(_xb7+0.029+_xr7)/2.0,_yE7
    fig.text(_lx7,_ly7,'T+6 h ('+('%+.1f'%_d7).replace('-',chr(8722))+' °C'+_sfx7+('' if _sfx7 else ')'),fontsize=FS_DENSE_SM,fontweight='bold',color='#555',ha='center',va='center',zorder=6,bbox=dict(facecolor='white',edgecolor='none',pad=1.2))
# blue-PI key REMOVED (user 2026-08-14): the caption formally defines the blue parenthesized ranges; on-figure key was redundant
# panel-c skill strapline REMOVED 2026-07-01 per user (declutter) -> skill (7-12%% vs DLinear / 57-61%% vs persistence, every cover) goes in caption
# panel-c conformal-coverage header REMOVED 2026-07-01 per user (decluttered); coverage claim -> Fig 10 caption   # n=10 6h coverage: per-station 86.7–95.4% (network 90.2%; correct 6h statistic, replaces the across-horizon 89.6–92.9% which stays in §4.5/Fig 8c). '(Attn-LSTM)' = the interval model; points are per-station best-of-4 → discloses the interval↔point mismatch (Option B).
if QUARTER=='Afternoon':   # coverage caveat for the panel-c intervals → bottom strip, afternoon only (the regime where the 90% PI under-covers); white bbox + raised GridSpec bottom keep it clear of the inset '% bldg' sublines
    pass  # afternoon ~80% under-coverage caveat REMOVED 2026-07-01 per user -> MANDATORY in Fig 10 caption   # N3 SLIMMED: short on-figure safety tag only; mechanism (under-forecasts peak ~12%, slope 0.876) + the header(all-hours 90%)/footer(afternoon ~80%) reconciliation → CAPTION
_fn='fig10_landcover_forecast'+('' if QUARTER=='Afternoon' else '_'+QUARTER.replace('-','').lower())   # PROMOTED 2026-08-05: preview design (source-area disks + Built n=2) is now the canonical Fig 10
savefig(fig,os.path.join(BASE,_fn),bbox='fullwidth'); plt.close()   # EXACT FULL_W per feedback_fullwidth_save (audit 2026-08-17); print('saved '+_fn)
