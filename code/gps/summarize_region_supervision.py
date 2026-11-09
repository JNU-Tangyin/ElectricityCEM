"""Summarize all Exp39 settings, OOF nulls and descriptive cluster membership."""
import os
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/energy-gps-mpl')
from pathlib import Path
import json,hashlib,base64,html
from itertools import combinations
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score,balanced_accuracy_score
from plot_gps_distance_clusters import align_to_reference
from gps_context import RESULTS_ROOT
from gps_region_supervision import predict, sample_triples

ROOT=RESULTS_ROOT/'region_supervision_exp39';SEEDS=['cem_seed0','cem_seed1','cem_seed42']

def holm(p):
 p=np.asarray(p);order=np.argsort(p);q=np.empty(len(p));q[order]=np.minimum(1,np.maximum.accumulate(p[order]*(len(p)-np.arange(len(p)))));return q

def table(d):return '<div class="scroll">'+d.to_html(index=False,border=0,float_format=lambda x:f'{x:.4f}',na_rep='—')+'</div>'
def image(path,caption):return '<figure><img src="data:image/png;base64,'+base64.b64encode(path.read_bytes()).decode()+'"><figcaption>'+html.escape(caption)+'</figcaption></figure>'

def main():
 scores=[];summaries=[];comparisons=[];clusterstats=[];clustercores=[];regions=[];sources={};training_fit=[]
 for k in [3,4,5]:
  for protocol in ['all','exclude_Rain']:
   folder=ROOT/f'k{k}_{protocol}';d=pd.read_csv(folder/'scores.csv');assert set(d.rep)==set(range(200));assert len(d)==1200
   scores.append(d);avg=d.groupby(['model','rep'])[['balanced_accuracy','macro_f1']].mean()
   for model in ['raw_centroid','triplet_mlp']:
    for metric in ['balanced_accuracy','macro_f1']:
     v=avg.loc[model,metric];obs=v.loc[0];null=v.loc[1:]
     summaries.append(dict(k=k,protocol=protocol,model=model,metric=metric,observed=obs,null_mean=null.mean(),null_q95=null.quantile(.95),p_upper=(1+(null>=obs).sum())/200,permutations=199))
   for metric in ['balanced_accuracy','macro_f1']:
    delta=avg.loc['triplet_mlp',metric]-avg.loc['raw_centroid',metric];null=delta.loc[1:]
    comparisons.append(dict(k=k,protocol=protocol,metric=metric,observed_gain=delta.loc[0],null_mean=null.mean(),null_q95=null.quantile(.95),p_upper=(1+(null>=delta.loc[0]).sum())/200))
   labels=pd.read_csv(folder/'cluster_labels.csv',dtype={'household':str});ids=sorted(labels.household.unique());pairs=np.triu_indices(len(ids),1)
   saved=np.load(folder/'embeddings.npz');saved_ids=saved['households'].tolist()
   details=pd.read_csv(folder/'oof_predictions.csv',dtype={'household':str})
   regmap=pd.read_csv(folder/'regions.csv').sort_values('coordinate').region.to_numpy()
   for seed in SEEDS:
    sub=details[(details.seed==seed)&(details.model=='triplet_mlp')].set_index('household').loc[saved_ids]
    city=sub.city.to_numpy();truth=sub.region.to_numpy();z=saved[seed];wt=1/np.bincount(city)[city]
    resub=predict(z[None],np.ones(len(z),bool),city,regmap[None],k)[0]
    trips=sample_triples(city,regmap,3999);a,p,n=trips.T
    order=(np.linalg.norm(z[a]-z[p],axis=1)<np.linalg.norm(z[a]-z[n],axis=1)).mean()
    training_fit.append(dict(k=k,protocol=protocol,seed=seed,resub_balanced_accuracy=balanced_accuracy_score(truth,resub,sample_weight=wt),training_triplet_order_accuracy=order))
   for method,g in labels.groupby('method'):
    labs={s:g[g.seed==s].set_index('household').loc[ids,'cluster'].to_numpy() for s in SEEDS}
    for a,b in combinations(SEEDS,2):
     x,y=labs[a],labs[b];cx=(x[pairs[0]]>=0)&(x[pairs[0]]==x[pairs[1]]);cy=(y[pairs[0]]>=0)&(y[pairs[0]]==y[pairs[1]])
     # Noise households are singleton groups, not one shared noise class.
     xx=np.where(x>=0,x,-1-np.arange(len(x)));yy=np.where(y>=0,y,-1-np.arange(len(y)))
     clusterstats.append(dict(k=k,protocol=protocol,method=method,seed_a=a,seed_b=b,ari=adjusted_rand_score(xx,yy),pair_jaccard=(cx&cy).sum()/(cx|cy).sum() if (cx|cy).sum() else np.nan))
    aligned,_,ncolors=align_to_reference(labs,SEEDS);matrix=np.column_stack([aligned[s] for s in SEEDS])
    for c in range(1,ncolors+1):
     inter=np.all(matrix==c,axis=1);union=np.any(matrix==c,axis=1)
     clustercores.append(dict(k=k,protocol=protocol,method=method,matched_cluster=c,**{s:int((aligned[s]==c).sum()) for s in SEEDS},intersection=int(inter.sum()),union=int(union.sum()),jaccard=float(inter.sum()/union.sum()),core_households=';'.join(np.array(ids)[inter])))
   reg=pd.read_csv(folder/'regions.csv');active=~reg.is_rain if protocol=='exclude_Rain' else np.ones(len(reg),bool)
   for region,g in reg[active].groupby('region'):regions.append(dict(k=k,protocol=protocol,region=int(region),coordinates=len(g),households=int(g.households.sum())))
   for p in folder.glob('*'):
    if p.is_file():sources[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
 summary=pd.DataFrame(summaries);gains=pd.DataFrame(comparisons)
 for metric in ['balanced_accuracy','macro_f1']:
  for model in ['raw_centroid','triplet_mlp']:
   mask=(summary.metric==metric)&(summary.model==model);summary.loc[mask,'p_holm_six_configs']=holm(summary.loc[mask,'p_upper'])
  mask=gains.metric==metric;gains.loc[mask,'p_holm_six_configs']=holm(gains.loc[mask,'p_upper'])
 clusterstats=pd.DataFrame(clusterstats);cores=pd.DataFrame(clustercores);regions=pd.DataFrame(regions)
 for name,df in [('summary',summary),('gain_vs_raw',gains),('cluster_seed_agreement',clusterstats),('cluster_core',cores),('region_support',regions),('full_panel_training_fit',pd.DataFrame(training_fit)),('all_scores',pd.concat(scores,ignore_index=True))]:df.to_csv(ROOT/f'{name}.csv',index=False)
 fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
 for ax,k in zip(axes,[3,4,5]):
  reg=pd.read_csv(ROOT/f'k{k}_all/regions.csv');ax.scatter(reg.lon,reg.lat,c=reg.region,cmap='tab10',s=15+reg.households*8,alpha=.8);rain=reg[reg.is_rain];ax.scatter(rain.lon,rain.lat,s=220,facecolors='none',edgecolors='black');ax.set(title=f'K={k} | 51 coordinate groups',xlabel='Longitude',ylabel='Latitude')
 fig.suptitle('GPS regions: unit-sphere KMeans | circle: Rain');fig.savefig(ROOT/'regions.png',dpi=160);plt.close(fig)
 fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained');saved=np.load(ROOT/'k4_all/embeddings.npz')
 lab=pd.read_csv(ROOT/'k4_all/cluster_labels.csv',dtype={'household':str});ids=saved['households'].tolist()
 for ax,s in zip(axes,SEEDS):
  xy=PCA(n_components=2).fit_transform(saved[s]);region=lab[(lab.seed==s)&(lab.method=='kmeans')].set_index('household').loc[ids,'region'];ax.scatter(xy[:,0],xy[:,1],c=region,cmap='tab10',s=25);ax.set(title=s,xlabel='PC1',ylabel='PC2')
 fig.suptitle('K=4 all | real-label full-panel embeddings | colors: target regions (not clusters)');fig.savefig(ROOT/'primary_pca.png',dpi=160);plt.close(fig)
 primary=summary.query("k==4 and protocol=='all' and model=='triplet_mlp' and metric=='balanced_accuracy'").iloc[0]
 primarygain=gains.query("k==4 and protocol=='all' and metric=='balanced_accuracy'").iloc[0]
 raw=summary.query("k==4 and protocol=='all' and model=='raw_centroid' and metric=='balanced_accuracy'").iloc[0]
 counts=pd.concat([pd.read_csv(ROOT/f'k{k}_{pr}/cluster_counts.csv') for k in [3,4,5] for pr in ['all','exclude_Rain']],ignore_index=True)
 counts.to_csv(ROOT/'cluster_counts.csv',index=False)
 contents='<h2>主配置结果</h2>'+f'<p>K=4、全部71户，三seed平均留出坐标balanced accuracy：三元组网络{primary.observed:.2%}，原始θ对照{raw.observed:.2%}；网络置换均值{primary.null_mean:.2%}，95%分位{primary.null_q95:.2%}，单侧p={primary.p_upper:.3f}，六配置Holm校正p={primary.p_holm_six_configs:.3f}。网络相对原始θ增量{100*primarygain.observed_gain:.2f}个百分点，其置换p={primarygain.p_upper:.3f}。</p>'
 contents+='''<h2>研究问题与预定方法</h2><p>以不同坐标、同区域家庭构造正样本，检验策略向量能否预测预先定义的粗地理区域。35维输入删除location权重、保留原bias；CEM策略未重训。K=4/all为主配置，K=3/5和exclude_Rain为敏感性分析；每个配置199次坐标到区域标签置换。</p><p>在全部51个唯一坐标的单位球xyz上做KMeans（n_init=50、seed=0），不按家庭数加权。两种协议共用这组区域定义，去掉Rain只改变样本。K=3/4/5在全部样本及去掉Rain后均至少8户/类；标签定义使用已知坐标，不使用θ或用电结果。</p><p>沿用Exp36的5折坐标分组。各折标准化只拟合训练家庭。三元组先抽坐标再抽家庭；正例必须同区域、不同坐标。网络35→64→64→64→64，前三层ReLU，欧氏三元组间隔1，SGD=.01，10轮；每次拟合固定256个三元组、批量32。结构和损失借鉴师兄代码，采样与留出协议为本实验改造，不称为URL原实验的完整复现。</p><p>用训练坐标等权的区域中心进行最近中心分类，原始标准化θ使用相同分类器。全部OOF预测拼接后，先按坐标等权计算balanced accuracy/macro-F1，再对三个seed均值。置换保持各区域坐标数和同坐标标签一致；每次重新训练映射与分类器。矩阵批量计算仅加速相互独立的网络，不共享拟合权重；与逐模型SGD已核对一致。</p>'''
 contents+='<h2>区域规模</h2>'+table(regions)+image(ROOT/'regions.png','点表示唯一记录坐标，面积反映家庭数；黑圈为Rain。')
 contents+='<h2>全部留出结果与置换参照</h2>'+table(summary)+'<h3>相对原始θ对照的增量</h3>'+table(gains)
 contents+='<h2>全样本拟合检查</h2><p>下表在真实区域标签的全样本拟合上计算，仅检查网络对训练任务的拟合程度，不是留出表现。</p>'+table(pd.DataFrame(training_fit))+'<h2>全样本聚类：描述性补充</h2><p>仅对真实区域标签用全样本拟合表示，再运行HDBSCAN和指定K的KMeans。下表不使用OOF嵌入；未为这些聚类指标建立null，不能把区域分类或监督后的图形直接视为家庭行为类型证据。PCA颜色为监督区域标签，非发现的簇。</p>'+table(counts)+table(clusterstats.groupby(['k','protocol','method'])[['ari','pair_jaccard']].mean().reset_index())+table(cores.query("k==4 and protocol=='all'"))+image(ROOT/'primary_pca.png','主配置三seed的独立PCA投影，颜色按目标区域显示。')
 contents+='''<h2>结果的适用范围</h2><p>区域划分由坐标构造，检验的是θ与这一粗地理目标的对应关系；区域不是用电类型标签。一个区域内可包含不同策略类型，跨区域也可能共享行为。即使留出分类超过null，仍需独立的时序与行为解释才能讨论家庭类型。</p><p>71户重复训练得到的三个seed不视为213户；样本实际覆盖约12天。全部6个配置及两种指标均报告，balanced accuracy为主指标，macro-F1为补充；Holm在每模型/指标的6配置中校正，增量检验另做6配置校正。199次置换的最小p分辨率为0.005。训练可用地点有限，坐标精度和既有θ的地点/缩放训练口径同前。</p>'''
 css='body{margin:0;background:#f3f5f8;color:#24364b;font:15px/1.8 -apple-system,BlinkMacSystemFont,"PingFang SC",sans-serif}header{background:#18344f;color:white;padding:35px}main{max-width:1200px;margin:25px auto;background:white;padding:30px}h2{color:#24517c;margin-top:35px}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:12px}td,th{padding:8px;border-bottom:1px solid #d9e2ec;white-space:nowrap;text-align:left}th{background:#eaf0f7}figure{margin:20px 0}img{max-width:100%}figcaption{font-size:13px;color:#52677d}@media(max-width:600px){main{padding:15px;margin:8px}header{padding:20px}}'
 page='<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Exp39 GPS区域监督实验</title><style>'+css+'</style></head><body><header><h1>Exp39 · GPS区域标签与三元组表示学习</h1><p>留出坐标检验 · 199次置换 · 2026-11-08</p></header><main>'+contents+'</main></body></html>'
 (ROOT/'report.html').write_text(page);(ROOT/'report_sources.json').write_text(json.dumps({'sources':sources,'primary':primary.to_dict(),'primary_gain':primarygain.to_dict()},indent=2))
 (ROOT/'README.md').write_text('# Exp39 GPS区域监督\n\n报告：report.html。主配置K=4/all/35维。固定协议见protocol_before_results.json；全部六配置见summary.csv，原始θ增量见gain_vs_raw.csv，逐户留出预测位于各配置目录。\n\n运行：python code/gps/run_region_suite.py；完成后 python code/gps/summarize_region_supervision.py。\n')
 print(summary.to_string(index=False));print('PRIMARY',primary.to_dict());print('GAIN',primarygain.to_dict())

if __name__=='__main__':main()
