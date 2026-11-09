"""Build factual, standalone 34D/36D comparison from complete saved GPS runs."""
from pathlib import Path
import base64
import hashlib
import html
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[3]
OLD=ROOT/'04.results/shape'
NEW=OLD/'gps_full36'
OUT=ROOT/'document/gps_full36_comparison_2026-11-08.html'
SEEDS=['cem_seed0','cem_seed1','cem_seed42']
SOURCES={}


def read(root,path):
    p=root/path
    SOURCES[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    return pd.read_csv(p)


def table(frame):
    return '<div class="scroll">'+frame.to_html(index=False,border=0,na_rep='—',float_format=lambda x:f'{x:.4f}')+'</div>'


def figure(path,caption):
    p=NEW/path
    SOURCES[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    return '<figure><img loading="lazy" src="data:image/png;base64,'+base64.b64encode(p.read_bytes()).decode()+'" alt="'+html.escape(caption)+'"><figcaption>'+html.escape(caption)+'</figcaption></figure>'


def paired(path,keys,metrics,select=None):
    a,b=read(OLD,path),read(NEW,path)
    if select:
        a,b=select(a),select(b)
    return a[keys+metrics].merge(b[keys+metrics],on=keys,suffixes=('_34D','_36D'),validate='one_to_one')


def label_summary():
    rows=[]
    for dim,root in [(34,OLD),(36,NEW)]:
        for exp,file,selector,model_col in [
            ('Exp33','exp33_gps_cluster_comparison/observed_labels.csv',lambda d:d[d.protocol=='all'],'arm'),
            ('Exp34','exp34_gps_network_discovery/labels.csv',lambda d:d,'model'),
            ('Exp35','exp35_place_networks/labels.csv',lambda d:d[d.protocol=='all'],'model'),
            ('Exp36','exp36_gps_distance_regression/labels.csv',lambda d:d,'model')]:
            d=selector(read(root,file))
            col='label' if 'label' in d else 'cluster'
            idcol='household' if 'household' in d else 'source'
            for (model,seed),g in d.groupby([model_col,'seed']):
                counts=g[g[col]>=0][col].value_counts()
                rows.append(dict(input_dims=dim,experiment=exp,model=model,seed=seed,k=len(counts),
                    sizes='/'.join(map(str,counts.sort_values(ascending=False).tolist())),noise=int((g[col]<0).sum())))
    return pd.DataFrame(rows)


def main():
    manifest=json.loads((NEW/'run_manifest.json').read_text())
    assert manifest['input_dimension']==36 and manifest['permutation_budgets']=={'exp33':99,'exp34':39,'exp35':39,'exp36':99}
    checks=[]
    expected=[('exp33_gps_cluster_comparison/cluster_coverage.csv','permutation',100),
              ('exp34_gps_network_discovery/clusters.csv','permutation',40),
              ('exp35_place_networks/cluster_metrics.csv','rep',40),
              ('exp36_gps_distance_regression/fold_metrics.csv','rep',100)]
    for path,column,count in expected:
        d=read(NEW,path)
        assert set(d[column])==set(range(count)),path
        checks.append(dict(check=path,expected_replicates=count,observed_replicates=d[column].nunique()))
    weights=read(NEW,'exp36_gps_distance_regression/weights.csv')
    assert set(weights.original_dim)==set(range(36))
    assert weights.groupby(['fold','seed']).size().eq(36).all()
    emb=np.load(NEW/'exp36_gps_distance_regression/embeddings.npz')
    assert all(emb[f'{s}_weighted'].shape==(71,36) for s in SEEDS)
    d32=paired('exp32_gps_triplet/summary.csv',['protocol','seed','metric'],['geographic_order_accuracy','neighbor_action_gap_kw'])
    d33=paired('exp33_gps_cluster_comparison/comparison.csv',['protocol','arm'],['observed_jaccard','null_mean','null_q95','valid_null'])
    d34=paired('exp34_gps_network_discovery/summary.csv',['model','metric'],['observed','null_mean','null_q95','p_upper','valid_null'])
    d35=paired('exp35_place_networks/summary.csv',['protocol','model','metric'],['observed','null_mean','null_q95','valid_null'])
    d36=paired('exp36_gps_distance_regression/cv_summary.csv',['model','split'],['mae_km','rmse_km'],lambda d:d[d.rep==0])
    p36=paired('exp36_gps_distance_regression/permutation_comparison.csv',['split'],['mae_improvement_km','null_mean','null_q95','p_upper'])
    sp=paired('exp36_gps_distance_regression/spectral_comparison/cross_seed_summary.csv',['model','n_neighbors','k'],['ari','pair_jaccard','matched_fraction'])
    d37=paired('exp37_gps_behavior_increment/summary.csv',['alpha','metric'],['constant_rmse','profiles_rmse','gps_rmse'])
    invariance=[]
    for file in ['behavior.csv','behavior_by_context.csv','input_profiles.csv','summary.csv','predictions.csv','stability.csv']:
        a=read(OLD,'exp37_gps_behavior_increment/'+file);b=read(NEW,'exp37_gps_behavior_increment/'+file)
        pd.testing.assert_frame_equal(a,b,check_exact=True)
        invariance.append(dict(file=file,rows=len(a),identical=True))
    ls=label_summary()
    spectra=read(NEW,'exp36_gps_distance_regression/spectral_comparison/cluster_sizes.csv')
    assert len(spectra)==54
    compare_dir=NEW/'comparison'
    compare_dir.mkdir(exist_ok=True)
    for name,df in [('exp32',d32),('exp33',d33),('exp34',d34),('exp35',d35),('exp36_cv',d36),('exp36_permutation',p36),('spectral',sp),('exp37',d37),('cluster_sizes',ls),('exp37_invariance',pd.DataFrame(invariance)),('completion_checks',pd.DataFrame(checks))]:
        df.to_csv(compare_dir/f'{name}_34_vs_36.csv',index=False)
    sections=[]
    def section(key,title,body):
        sections.append((key,title,'<section id="'+key+'"><h2>'+title+'</h2>'+body+'</section>'))
    quick=[]
    for exp,model,label in [('Exp34','raw','原始θ'),('Exp34','diagonal','Exp34 地理排序对角映射'),('Exp34','network','Exp34 地理排序网络'),('Exp35','supcon','Exp35 同地点SupCon'),('Exp35','triplet','Exp35 同地点三元组'),('Exp36','weighted','Exp36 连续距离回归')]:
        row={'分支':label}
        for dim in [34,36]:
            d=ls[(ls.input_dims==dim)&(ls.experiment==exp)&(ls.model==model)].set_index('seed')
            row[f'{dim}维簇数（seed 0／1／42）']='／'.join(str(int(d.loc[s,'k'])) for s in SEEDS)
        quick.append(row)
    section('overview','概览 · 主要变化',table(pd.DataFrame(quick))+'''<p>原始θ小簇的三个seed成员名单与34维版本相同，共同成员仍为10408与10495。Exp35 SupCon三个seed的共同成员全部集中在同一个匹配簇中，为Rain的18户，其余匹配簇的三次交集均为0。</p>
    <p>Exp35同地点三元组的家庭对Jaccard由0.1013变为0.3655；36维版本的主要匹配簇有29户三次共同成员，其中18户为Rain、11户来自其他地点。各seed的另一小簇无共同家庭。</p>
    <p>Exp36留出地点到训练地点的距离MAE由117.9242 km变为117.6515 km；常数对照为113.3398 km。相对等权距离改善的置换p值由0.13变为0.14。Exp37重复运行结果逐项相同。</p>''')
    section('protocol','01 · 实验设置','''<p>将映射输入从去掉location权重和bias的34维，改为完整36维θ。location仍是联邦州编码输入对应的策略权重，bias为原策略偏置。CEM策略文件、71户样本、三个CEM seed（0、1、42）、GPS坐标、训练划分及超参数沿用原实验。</p>
    <p>Exp32–36重新执行监督映射和聚类；Exp33/36保留99次置换，Exp34/35保留39次置换。Exp33/35/32同时运行全部71户和不含Rain的53户补充协议。谱聚类重新运行54个组合。Exp37重新执行原行为预测流程。</p>
    <p>Exp34三元组网络输入层改为36→64→64→64→64；Exp35 SupCon为36→32→8。对角映射保留36个非负权重；Exp36距离归一化分母随输入维数改为36。HDBSCAN参数仍为min_cluster_size=4、min_samples=2。谱聚类为K=2/3/4，邻居数5/10/15。</p>
    <p>GPS监督与行为定义保持一致：共同场景回放采用宿主完整287小时数据的缩放和地点输入，回放末段72小时；各策略独立更新SOC。以下表格中34D列取自原结果，36D列取自本轮运行。</p>''')
    section('sizes','02 · 聚类数量与分配规模',table(ls)+'''<p>sizes为各已分配簇的户数，按从大到小排列；noise为未归簇户数。簇数取自实际输出。</p>''')
    d32mean=d32.groupby(['protocol','metric']).mean(numeric_only=True).reset_index()
    section('exp32','03 · Exp32：留地点地理排序',table(d32mean)+'''<p>三seed均值；近邻动作距离单位kW。protocol=all为71户，exclude_largest为不含Rain的53户。</p>''')
    section('exp33','04 · Exp33：监督形式对照',table(d33)+'''<p>observed_jaccard为三个seed配对的同簇家庭对Jaccard均值。null_q95为置换分布95%分位；valid_null为满足既有汇总条件的有效置换数。</p>''')
    section('exp34','05 · Exp34：地理排序映射',table(d34[d34.metric.isin(['train','test','jaccard','gap_kw'])])+figure('exp34_gps_network_discovery/cluster_visualization/diagonal/pca_three_seeds.png','Exp34，36维输入，对角映射；PCA仅展示。')+figure('exp34_gps_network_discovery/cluster_visualization/network/pca_three_seeds.png','Exp34，36维输入，三元组网络；PCA仅展示。'))
    body=table(d35[d35.protocol=='all'])
    for model in ['raw','supcon','triplet']:
        prefix=f'exp35_place_networks/cluster_visualization/{model}/'
        cores=read(NEW,prefix+'matched_cluster_overlap.csv')
        body+='<h3>'+html.escape(model)+'：36维输入的跨seed成员交集</h3>'+table(cores)+figure(prefix+'pca_three_seeds.png',f'Exp35，36维输入，{model}。颜色按最大成员交集匹配，灰色为未归簇。')
    body+='<details><summary>53户补充协议</summary>'+table(d35[d35.protocol!='all'])+'</details>'
    section('exp35','06 · Exp35：同地点表示学习',body)
    section('exp36','07 · Exp36：连续地理距离回归',table(d36)+table(p36)+'''<p>MAE和改善量单位km；p_upper为原定义的单侧置换p值。</p>'''+table(read(NEW,'exp36_gps_distance_regression/cluster_visualization/weighted/matched_cluster_overlap.csv'))+figure('exp36_gps_distance_regression/cluster_visualization/weighted/pca_three_seeds.png','Exp36，36维加权表示与HDBSCAN。'))
    core_spectral=read(NEW,'exp36_gps_distance_regression/spectral_comparison/cluster_core.csv')
    section('spectral','08 · 谱聚类对照',table(sp)+'''<h3>36维输入：K=2的匹配少数组</h3>'''+table(core_spectral[(core_spectral.k==2)&(core_spectral.aligned_cluster==2)])+figure('exp36_gps_distance_regression/spectral_comparison/k2_neighbors10_pca.png','36维输入，K=2、邻居参数10：原始θ与Exp36表示。')+figure('exp36_gps_distance_regression/spectral_comparison/all_settings_stability.png','36维输入，全部18组空间/参数设置的跨seed指标。'))
    section('exp37','09 · Exp37：行为预测重复运行',table(d37[d37.alpha==10])+'''<p>Exp37直接使用完整策略回放行为，家庭特征加GPS的岭回归不接收34维或36维θ作为回归输入。本轮重新运行的行为、预测及汇总表与原结果逐项相同。</p>'''+table(pd.DataFrame(invariance)))
    a38=json.loads((NEW/'exp38_raw_small_cluster_audit/audit.json').read_text())
    data38=read(NEW,'exp38_raw_small_cluster_audit/all_households_data.csv')
    params38=read(NEW,'exp38_raw_small_cluster_audit/all_households_parameters_training.csv')
    section('exp38','10 · 36维原始θ的小簇检查',f'<p>三次小簇并集为{a38["candidate_n"]}户；三次共同成员为{a38["core_n"]}户：'+html.escape('、'.join(a38['core']))+'。</p>'+table(data38[data38.candidate][['household','minority_seed_count','city','zero_load_fraction','pv_peak_wh','complete_day_load_mean_kwh']])+table(read(NEW,'exp38_raw_small_cluster_audit/small_cluster_overlap.csv'))+f'<p>全部213个策略重新进行原家庭训练/验证回放；验证收益与日志最大绝对误差为{params38.val_replay_abs_error.max():.2e} EUR/h。保存策略最大绝对参数为{params38.max_abs_weight.max():.6f}，边界±1.5。数据描述指标在确认原文件哈希一致后复用，36维成员、质心距离和参考组百分位重新计算。</p>')
    links=[]
    for p in sorted(compare_dir.glob('*.csv')):
        links.append('<li><a download="'+p.name+'" href="data:text/csv;base64,'+base64.b64encode(p.read_bytes()).decode()+'">'+html.escape(p.name)+'</a></li>')
    section('files','11 · 结果与执行核对',table(pd.DataFrame(checks))+'<p>本轮结果根目录：<code>04.results/shape/gps_full36/</code>；原34维结果保留在<code>04.results/shape/</code>各实验目录。日志位于本轮根目录的logs/，执行入口为run_full36_suite.py；36维候选户检查入口为audit_full36_candidates.py。</p><ul>'+''.join(links)+'</ul>')
    css='''*{box-sizing:border-box}body{margin:0;background:#f3f5f8;color:#1b2c41;font:15px/1.8 -apple-system,BlinkMacSystemFont,"PingFang SC",sans-serif}header{background:#142f4d;color:white;padding:45px max(24px,calc((100vw - 1150px)/2))}h1{font-size:32px;line-height:1.4}header p{color:#d7e2ef}nav{background:white;border-bottom:1px solid #dde4ed;position:sticky;top:0;z-index:3;overflow:auto;white-space:nowrap;padding:12px 24px}nav a{margin-right:20px;color:#254f7e;text-decoration:none}main{max-width:1250px;margin:auto;padding:20px}section{scroll-margin-top:65px;background:white;padding:30px;margin:20px 0;border:1px solid #dce4ed;border-radius:8px}h2{font-size:24px;color:#214e7c}h3{font-size:19px;margin-top:30px}.scroll{overflow-x:auto;margin:20px 0}table{border-collapse:collapse;width:100%;font-size:12px;line-height:1.6}th,td{padding:10px;border-bottom:1px solid #dde4ed;text-align:left;white-space:nowrap}th{background:#eaf0f7}tr:nth-child(even){background:#f8fafc}figure{margin:25px 0}img{width:100%;height:auto}figcaption{font-size:13px;color:#596c81}summary{cursor:pointer;color:#245582}code{overflow-wrap:anywhere}footer{padding:25px;text-align:center;color:#596c81}@media(max-width:600px){main{padding:10px}section{padding:16px}h1{font-size:26px}h2{font-size:21px}}@media print{nav{display:none}body{background:white}section{break-inside:auto}figure{break-inside:avoid}main{padding:0}}'''
    content='<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>GPS完整36维重跑与34维对照</title><style>'+css+'</style></head><body><header><h1>GPS 完整36维重跑<br>与原34维结果对照</h1><p>71户 · CEM seed 0／1／42 · Exp32–38及谱聚类 · 2026-11-08</p><p>保留location权重与bias；其余实验协议沿用原设置。</p></header><nav>'+''.join('<a href="#'+key+'">'+title.split(' · ')[0]+'</a>' for key,title,_ in sections)+'</nav><main>'+''.join(body for _,_,body in sections)+'</main><footer>数值由保存结果生成；图像与对照CSV嵌入本文件。</footer></body></html>'
    OUT.write_text(content)
    (OUT.with_suffix('.sources.json')).write_text(json.dumps({'sources':SOURCES,'builder_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'report':str(OUT)},indent=2))
    (compare_dir/'README.md').write_text('# 34维与36维GPS实验对照\n\n完整报告：'+str(OUT)+'\n\n所有对照表按同一实验、协议、seed或模型键合并。原始结果未覆盖。Exp37复跑全部关键表逐项一致。\n')
    print('Saved',OUT,OUT.stat().st_size,'bytes')


if __name__=='__main__':
    main()
