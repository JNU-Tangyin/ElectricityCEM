"""Build the standalone 34/35/36D comparison from saved experiment tables."""
from pathlib import Path
import base64
import hashlib
import html
import json
import pandas as pd
import numpy as np

PACKAGE=Path(__file__).resolve().parents[2]
BASE=PACKAGE/'results/gps'
SEEDS=['cem_seed0','cem_seed1','cem_seed42']
SOURCES={}


def read(dim,rel):
    path=BASE/f'{dim}d'/rel
    SOURCES[str(path.relative_to(PACKAGE))]=hashlib.sha256(path.read_bytes()).hexdigest()
    return pd.read_csv(path)


def table(d):
    return '<div class="scroll">'+d.to_html(index=False,border=0,na_rep='—',float_format=lambda x:f'{x:.5f}')+'</div>'


def figure(rel,caption):
    path=BASE/'35d'/rel
    SOURCES[str(path.relative_to(PACKAGE))]=hashlib.sha256(path.read_bytes()).hexdigest()
    return '<figure><img src="data:image/png;base64,'+base64.b64encode(path.read_bytes()).decode()+'" alt="'+html.escape(caption)+'"><figcaption>'+html.escape(caption)+'</figcaption></figure>'


def main():
    rows=[];sizes=[];checks=[]
    def add(dim,exp,protocol,model,metric,value):
        rows.append(dict(dimension=dim,experiment=exp,protocol=protocol,model=model,metric=metric,value=value))
    for dim in [34,35,36]:
        d=read(dim,'exp32_gps_triplet/summary.csv')
        for (pr,m),g in d.groupby(['protocol','metric']):
            for field in ['geographic_order_accuracy','neighbor_action_gap_kw']:add(dim,'Exp32',pr,m,field,g[field].mean())
        d=read(dim,'exp33_gps_cluster_comparison/comparison.csv')
        for r in d.itertuples():
            for field in ['observed_jaccard','null_q95','valid_null']:add(dim,'Exp33',r.protocol,r.arm,field,getattr(r,field))
        d=read(dim,'exp34_gps_network_discovery/summary.csv')
        for r in d.itertuples():
            for field in ['observed','null_mean','null_q95','p_upper','valid_null']:add(dim,'Exp34','all',r.model,r.metric+'__'+field,getattr(r,field))
        d=read(dim,'exp35_place_networks/summary.csv')
        for r in d.itertuples():
            for field in ['observed','null_mean','null_q95','valid_null']:add(dim,'Exp35',r.protocol,r.model,r.metric+'__'+field,getattr(r,field))
        d=read(dim,'exp36_gps_distance_regression/cv_summary.csv')
        for r in d[d.rep==0].itertuples():add(dim,'Exp36',r.split,r.model,'mae_km',r.mae_km)
        d=read(dim,'exp36_gps_distance_regression/permutation_comparison.csv')
        for r in d.itertuples():
            for f in ['mae_improvement_km','null_q95','p_upper']:add(dim,'Exp36',r.split,'weighted_vs_raw',f,getattr(r,f))
        d=read(dim,'exp36_gps_distance_regression/spectral_comparison/cross_seed_summary.csv')
        assert len(d)==18
        for r in d.itertuples():
            for f in ['ari','pair_jaccard']:add(dim,'Spectral',f'K={r.k},neighbors={r.n_neighbors}',r.model,f,getattr(r,f))
        for exp,path in [('Exp34','exp34_gps_network_discovery/labels.csv'),('Exp35','exp35_place_networks/labels.csv'),('Exp36','exp36_gps_distance_regression/labels.csv')]:
            d=read(dim,path)
            if 'protocol' in d:d=d[d.protocol=='all']
            for (m,s),g in d.groupby(['model','seed']):
                counts=g[g.cluster>=0].cluster.value_counts()
                sizes.append(dict(dimension=dim,experiment=exp,model=m,seed=s,k=len(counts),sizes='/'.join(map(str,counts.sort_values(ascending=False))),noise=int((g.cluster<0).sum()),coverage=float((g.cluster>=0).mean()),smallest_cluster=int(counts.min()) if len(counts) else 0))
        for path,col,n in [('exp33_gps_cluster_comparison/cluster_coverage.csv','permutation',100),('exp34_gps_network_discovery/clusters.csv','permutation',40),('exp35_place_networks/cluster_metrics.csv','rep',40),('exp36_gps_distance_regression/fold_metrics.csv','rep',100)]:
            d=read(dim,path);assert set(d[col])==set(range(n)),(dim,path)
            checks.append(dict(dimension=dim,file=path,replicates=n))
    data=pd.DataFrame(rows)
    wide=data.pivot(index=['experiment','protocol','model','metric'],columns='dimension',values='value').reset_index()
    wide.columns=[f'{c}D' if isinstance(c,int) else c for c in wide.columns]
    sizes=pd.DataFrame(sizes)
    invariance=[]
    for file in ['behavior.csv','behavior_by_context.csv','input_profiles.csv','summary.csv','predictions.csv']:
        baseline=read(34,'exp37_gps_behavior_increment/'+file)
        for dim in [35,36]:
            pd.testing.assert_frame_equal(baseline,read(dim,'exp37_gps_behavior_increment/'+file),check_exact=True)
        invariance.append(dict(file=file,rows=len(baseline),identical_across_dimensions=True))
    w=read(35,'exp36_gps_distance_regression/weights.csv')
    assert set(w.original_dim)==set(range(36))-{3} and w.groupby(['fold','seed']).size().eq(35).all()
    e=np.load(BASE/'35d/exp36_gps_distance_regression/embeddings.npz')
    assert all(e[f'{s}_weighted'].shape==(71,35) for s in SEEDS)
    out=BASE/'comparison';out.mkdir(exist_ok=True)
    for name,d in [('metrics_34_35_36',wide),('cluster_sizes_34_35_36',sizes),('completion_checks',pd.DataFrame(checks)),('exp37_invariance',pd.DataFrame(invariance))]:d.to_csv(out/f'{name}.csv',index=False)
    sections=[]
    def section(key,title,body):sections.append((key,title,f'<section id="{key}"><h2>{title}</h2>{body}</section>'))
    quick=[]
    for exp,m,label in [('Exp34','raw','原始θ'),('Exp34','diagonal','地理排序对角映射'),('Exp34','network','地理排序网络'),('Exp35','supcon','同地点SupCon'),('Exp35','triplet','同地点三元组'),('Exp36','weighted','连续距离回归')]:
        row={'分支':label}
        for dim in [34,35,36]:
            d=sizes[(sizes.dimension==dim)&(sizes.experiment==exp)&(sizes.model==m)].set_index('seed')
            row[f'{dim}维：簇数；覆盖率；最小簇']='／'.join(str(int(d.loc[s,'k'])) for s in SEEDS)+'；'+'／'.join(f'{d.loc[s,"coverage"]:.0%}' for s in SEEDS)+'；'+'／'.join(str(int(d.loc[s,'smallest_cluster'])) for s in SEEDS)
        quick.append(row)
    # Audit identical spectral summary values using the saved household partitions.
    spectral_checks=[]
    for da,ma,db,mb in [(34,'raw_theta',36,'raw_theta'),(35,'raw_theta',35,'exp36_weighted')]:
        aa=read(da,'exp36_gps_distance_regression/spectral_comparison/labels.csv')
        bb=read(db,'exp36_gps_distance_regression/spectral_comparison/labels.csv')
        for seed in SEEDS:
            a=aa[(aa.model==ma)&(aa.n_neighbors==5)&(aa.k==2)&(aa.seed==seed)].sort_values('household')
            b=bb[(bb.model==mb)&(bb.n_neighbors==5)&(bb.k==2)&(bb.seed==seed)].sort_values('household')
            np.testing.assert_array_equal(a.household,b.household)
            x=a.cluster_original.to_numpy();y=b.cluster_original.to_numpy()
            same=bool(np.array_equal(x[:,None]==x[None,:],y[:,None]==y[None,:]))
            assert same
            spectral_checks.append(dict(comparison=f'{da}D {ma} vs {db}D {mb}',seed=seed,k=2,n_neighbors=5,identical_partition=same))
    spectral_checks=pd.DataFrame(spectral_checks)
    spectral_checks.to_csv(out/'spectral_identical_partition_audit.csv',index=False)
    positive=[];all_tests=[]
    for dim in [34,35,36]:
        dd=read(dim,'exp35_place_networks/summary.csv').query("model != 'raw'").copy()
        dd['dimension']=dim;all_tests.append(dd)
        positive.append(dd[dd.metric.isin(['jaccard','load_gap','action_gap_kw']) & (dd.observed>dd.null_q95)])
    positive=pd.concat(positive,ignore_index=True);all_tests=pd.concat(all_tests,ignore_index=True)
    positive.to_csv(out/'exp35_local_positive_results.csv',index=False)
    section('summary','摘要与主要结论','''<p><strong>Exp32–38未检测到在已运行设置间一致复现的跨地点策略类型。</strong>三种维度下，Exp34地理排序在留出地点均未超过地图置换参照；Exp36相对等权距离的改善均未通过单侧5%置换检验，且加权表示和等权表示的MAE均高于常数距离预测。</p><p>同地点SupCon能高度拟合标签，但随机标签也能被近乎完全拟合；跨seed复现主要集中在受监督的Rain家庭。36维同地点三元组在全71户协议中出现家庭对Jaccard和负荷差距的局部正结果，未在各维度与不含Rain协议中一致复现。34／35／36维对照没有改变总体判断，但明显改变了部分聚类划分。</p><p>动作回放用于描述策略分组，不作为独立验证。主体汇总重点检验与数据边界，方法、全量对照及局部正结果列入附录。</p>''')
    section('guidance','研究问题与GPS监督设计','''
    <p>本组实验围绕家庭策略表示的结构发现展开：利用地理信息辅助组织已训练的策略向量θ，观察新表示中是否出现跨训练seed可复现的家庭分组，再把分组对应回家庭负荷、PV及充放电行为，识别可能跨地点共有的策略类型。</p>
    <p>本组实验检验的技术路径包括“用GPS作为标签进行监督对比学习”“处理GPS的地理距离”“修改损失函数，或把样本间距离计算出来作为标签”。本次实验将这些路径落实为两类监督任务：同地点关系监督，以及连续地理距离／距离排序监督。两类任务分别报告，检验对象各有侧重。</p>
    <div class="pipeline"><span>已训练的策略θ</span><b>→</b><span>标准化输入x</span><b>→</b><span>学习映射f<sub>φ</sub></span><b>→</b><span>新表示z</span><b>→</b><span>聚类与跨seed核对</span><b>→</b><span>时序与策略行为解释</span></div>
    <p class="signal">GPS经纬度 → 同地点关系、球面距离或远近排序 → 构造监督损失 → 更新映射参数φ</p>
    <p>在Exp32–36中，GPS用于规定训练约束，映射的输入仍然是θ。新表示为z=f<sub>φ</sub>(x)，不是把经纬度直接拼接到θ后聚类。GPS也未被设置成聚类中心，聚类算法在映射完成后独立运行。既有CEM策略沿用保存的参数，本轮学习的是表示映射。</p>
    <h3>两种GPS监督分别回答什么</h3>
    <div class="scroll"><table><thead><tr><th>监督概念</th><th>实验中的实现</th><th>所检验的问题</th></tr></thead><tbody>
    <tr><td>GPS作为标签</td><td>相同记录坐标定义相同地点；构造同地点家庭正样本</td><td>地点关系能否指导θ形成可复现的新分组？</td></tr>
    <tr><td>处理地理距离</td><td>经纬度经Haversine转换为球面公里距离</td><td>θ及其映射能否反映真实地点间的远近关系？</td></tr>
    <tr><td>修改对比损失</td><td>以近地点为正例、远地点为负例构造三元组</td><td>地理远近排序能否从θ中学习，并推广到留出地点？</td></tr>
    <tr><td>样本间距离作为标签</td><td>家庭对(i,j)的地理距离D<sub>ij</sub>作为连续监督目标</td><td>表示距离能否拟合公里距离，且在留出地点保持关系？</td></tr>
    </tbody></table></div>
    <p>同地点任务只使用地点是否相同；距离任务额外使用不同地点之间的远近。连续距离标签属于“家庭对”，并非给每户指定一个类别编号。地理邻近与策略相似之间的关系是本组实验检验的假设。</p>
    <h3>GPS信号的数据构成</h3>
    <p>71户覆盖51个不同记录坐标组：Rain18户，Heilbronn、Berlin、Maselheim各2户，另外47组各1户。同坐标家庭对共156对，其中Rain占153对。地点监督的重复家庭来自这4个坐标组；47个单户地点可进入对比项，但没有同地点的另一户可构成正样本。主分析保留Rain，并单列不含Rain的53户补充协议。</p>
    <p>本面板中，同一地名的家庭共享一对记录坐标，因此同地点监督使用的是记录地点的一致性。坐标表没有提供同一城市内部家庭之间的位置差别。报告分别列出Rain与其他地点的共同成员，便于核对监督信号与所得分组的对应。</p>''')
    section('mapping','新空间如何形成：映射与损失函数','''
    <p>映射f<sub>φ</sub>通过优化GPS监督损失得到。训练开始时，给定一组初始维度权重或神经网络参数；训练不断调整φ，使新空间中的样本关系更符合地理约束。家庭的实际用电时序未被重新编码输入这些映射，θ作为既有策略的参数表示进入训练。</p>
    <h3>路线A：同地点监督对比学习（Exp35）</h3>
    <p>对于锚点家庭i，使用同一记录地点的另一户j作为正样本，其他地点家庭进入对比项。SupCon提高正样本在对比集合中的相对相似度。网络为d→32→8，隐藏层ReLU，输出单位化；温度0.1。d分别取34、35、36。训练后，HDBSCAN在8维z空间中进行聚类。</p>
    <p>该路线还运行同地点三元组网络：锚点a和正例p来自同地点且为不同家庭，负例n来自另一地点。损失为max(0, 1＋‖z<sub>a</sub>−z<sub>p</sub>‖−‖z<sub>a</sub>−z<sub>n</sub>‖)。网络结构为d→64→64→64→64，前三层ReLU。两种网络共用同地点定义，但损失形式、输出维度和训练设置不同。</p>
    <h3>路线B：地理距离排序（Exp32–34）</h3>
    <p>对锚点a，若球面地理距离D<sub>ap</sub>小于D<sub>an</sub>，则约束表示中p比n更接近a。此时三元组的正例可以来自另一个地点；“正例”表示相对更近，而非必须同地点。经纬度先转换为球面距离，避免直接把角度差作为公里距离。</p>
    <p>Exp32使用非负对角度量；Exp33对照同地点、地理排序及各半混合的三元组；Exp34比较对角度量与64维三元组网络。对角映射逐维调整标准化θ的权重，排序损失采用带正则的平方铰链形式；网络使用上述欧氏距离间隔损失。排序约束强调谁更近，不要求表示距离等于实际公里数。</p>
    <h3>路线C：直接以地理距离监督（Exp36）</h3>
    <p>对每个家庭对，计算连续标签D<sub>ij</sub>。学习非负权重w，形成z<sub>ik</sub>=x<sub>ik</sub>√(w<sub>k</sub>/d)，并以ŝ<sub>ij</sub>=s‖z<sub>i</sub>−z<sub>j</sub>‖预测地理公里距离。s由训练地点对的距离中位数确定。</p>
    <p>优化目标为地点对平衡的距离平方误差，加上向校准等权权重收缩的正则项（系数0.1）。该映射没有神经网络与激活函数。拟合只使用不同坐标家庭对，保留全部家庭；每折的标准化、尺度和权重都由训练地点确定，再评估留出地点。它具体实现了“把样本之间的距离直接算出来作为标签”的连续目标方案。</p>
    <h3>34／35／36维对照的位置</h3>
    <p>原θ的location项是联邦州编码对应的策略权重，GPS经纬度另行读取。34维去掉location权重及bias；35维只去掉location权重并保留原bias；36维保留全部参数。改变这两维会改变输入的几何关系，因此在同一套GPS损失与聚类设置下分别运行。三种维数的对照考察输入表示选择，GPS监督的定义保持一致。</p>''')
    section('evaluation','从GPS拟合到类型发现：评价流程','''
    <p>评价沿着“监督目标拟合 → 留出地点表现 → 聚类成员复现 → 行为关联”展开。拟合GPS约束说明模型完成了指定训练任务；跨seed成员交集说明同一家庭能否在不同CEM训练下保持分组；时序和动作差异用于描述这些分组的实际含义。</p>
    <div class="scroll"><table><thead><tr><th>检查环节</th><th>指标或对照</th><th>解释对象</th></tr></thead><tbody>
    <tr><td>监督目标拟合</td><td>同地点排序正确率／地理排序正确率／距离MAE</td><td>新空间满足所给GPS约束的程度</td></tr>
    <tr><td>留出地点检验</td><td>Exp32逐地点留出；Exp34/36按地点分5折</td><td>θ→地理关系的映射在未参与拟合地点的表现</td></tr>
    <tr><td>置换对照</td><td>标签对应置换或整地点坐标置换后重新拟合</td><td>真实GPS对应关系相对随机对应提供的增量</td></tr>
    <tr><td>成员复现</td><td>三个CEM seed的逐簇交集、并集及家庭对Jaccard</td><td>分组是否反复包含同一批家庭</td></tr>
    <tr><td>回到行为</td><td>24小时负荷形状、共同场景充放电动作、Exp37行为预测</td><td>分组对应何种输入特征或策略行为</td></tr>
    </tbody></table></div>
    <p>Exp33、Exp35置换家庭与地点标签的对应关系，保持地点标签的数量；Exp34、Exp36保留同地点家庭组，重排地点组的坐标。两类零参照各自对应原监督问题。报告同时给出观测值和置换分布，地理排序的比较使用实际置换水平。</p>
    <p>家庭对Jaccard汇总所有同簇家庭对，大簇贡献较多；逐簇交集进一步显示小簇和主体簇各自的成员复现。HDBSCAN簇数由结果产生；谱聚类的K=2、3、4是输入设置。PCA图用于显示完整表示的投影，成员核对依据实际高维聚类标签。</p>
    <p>共同场景动作由同一批θ回放产生，只用于解释参数分组的行为含义，不作为独立验证证据。已有动作差距指标是簇间与簇内的平均差，未逐簇命名充放电类型。负荷时序关联及跨seed核对与之并列报告。Exp37另行检验家庭特征加入GPS后，对策略行为描述符的留地点预测是否改变。</p>''')
    section('overview','三种输入的对照',table(pd.DataFrame(quick))+'''<p>34维删除location权重与bias；35维只删除location权重，保留原始bias；36维保留全部参数。35维没有将location贡献合并进bias。表中簇数、覆盖率和最小簇规模均依次对应CEM seed 0／1／42。完整逐seed表可下载。</p><p>样本为相同71户，主分析保留Rain18户；已有53户补充协议继续运行。CEM权重文件不变，仅改变GPS表示学习的输入维数。Exp33/36各99次置换，Exp34/35各39次置换，谱聚类每种维度54次拟合。</p>
    <p>35维SupCon的簇数为4／5／4：一个匹配簇的三次交集为Rain的18户，另一个匹配簇的交集为6609，其余匹配簇交集为0。35维同地点三元组的簇数为2／4／2，主要匹配簇的三次交集为15户，全部来自Rain。</p>
    <p>同地点SupCon的家庭对Jaccard在34／35／36维下依次为0.351／0.373／0.360；同地点三元组依次为0.101／0.232／0.365。原始θ小簇的三次共同成员在三种输入下均为10408、10495。</p>''')
    alignment_note_shown=False
    for exp in ['Exp32','Exp33','Exp34','Exp35','Exp36','Spectral']:
        d=wide[wide.experiment==exp].drop(columns='experiment')
        # Preserve every metric in CSV, show observed results and null thresholds in the main table.
        main=d[~d.metric.str.endswith('__null_mean')]
        descriptions={
            'Exp32':'逐地点留出检验地理排序监督的对角映射，并比较新近邻策略的动作距离。geographic_order_accuracy为地理远近排序正确率，neighbor_action_gap_kw为最近5户的平均动作距离；表中对三个seed取均值。',
            'Exp33':'在相同三元组预算和对角映射目标下，比较同地点标签（place_id）、地理排序（distance）和各半混合（mixed）。observed_jaccard为跨seed家庭对Jaccard，null_q95为标签置换对照的95%分位。',
            'Exp34':'比较非负对角映射和神经网络能否学习地理远近排序。train和test分别为训练与留出地点排序正确率；jaccard衡量全样本聚类的成员复现。p_upper由整地点坐标置换得到。',
            'Exp35':'同地点GPS标签用于学习新空间；place_fit表示同地点样本相对异地点更近的正确率，coverage为已归簇比例。action_gap_kw和load_gap分别是簇间距离减簇内距离；正值表示按该指标计，同簇家庭平均更接近。',
            'Exp36':'连续公里距离作为监督目标。mae_km为距离预测平均绝对误差；test_to_train对应留出地点与训练地点之间，test_to_test对应留出地点之间。constant为训练地点对平均距离，raw_scaled为校准等权表示，weighted为GPS监督加权表示。',
            'Spectral':'固定映射得到的表示后改变聚类器及构图参数，检查分组对算法设置的敏感性。raw_theta为原始标准化输入，exp36_weighted为连续距离监督表示；ARI和家庭对Jaccard均在三个seed配对间汇总。'}
        notes={
            'Exp32':'本节为早期探索，未实施置换null。候选家庭对的排序正确率未按候选地点对等权；最终先在留出坐标组内汇总，再对坐标组取均值。正式地理排序判断采用Exp34的地点平衡与地图置换结果，不能用0.5单独判断本节结果。',
            'Exp34':'raw未实施聚类null。原始θ在三个seed中包含64–66户的主体簇与4–7户的小簇；整体家庭对Jaccard约0.816–0.833主要反映主体簇重合。其小簇三次交并比为2/10=0.200。聚类复现的解读以逐簇成员交集与谱聚类ARI并列核对。',
            'Exp35':'place_fit属于训练标签拟合，不等同于发现家庭类型；监督构造的Rain集中也不是独立验证。valid_null为实际有效置换数，各指标可能不同。action_gap仅描述同一批θ的回放差异，不能独立验证由θ产生的簇。',
            'Exp36':'在留出→训练地点评估中，34/35/36维的constant MAE均为113.340 km，weighted分别为117.924/117.655/117.652 km，raw_scaled分别为121.554/121.244/121.240 km。两种表示都比常数预测差；相对等权表示改善不等于超过常数基准。当前Exp36预测为s√(design·w)，没有独立常数项；这一模型不包含任意非零常数距离作为可单独拟合的特例。与constant的比较是预测误差对照，不是嵌套模型检验。含常数项的距离模型尚未运行。',
            'Spectral':'本节无聚类null，K为预设值。K=2在部分设置中切出少数家庭，不能据此称为两个类型。raw 34维、K=2、邻居5时，平均ARI仅0.01828而家庭对Jaccard为0.73474，显示整体配对重合可能由大簇主导。'}
        body='<p>'+descriptions[exp]+'</p>'+('<p class="note">'+notes[exp]+'</p>' if exp in notes else '')+table(main)
        if exp in ['Exp34','Exp35','Exp36']:
            specs={'Exp34':[('exp34_gps_network_discovery/cluster_visualization/diagonal','对角映射'),('exp34_gps_network_discovery/cluster_visualization/network','地理排序网络')],
                   'Exp35':[(f'exp35_place_networks/cluster_visualization/{m}',m) for m in ['raw','supcon','triplet']],
                   'Exp36':[('exp36_gps_distance_regression/cluster_visualization/weighted','连续距离回归')]}[exp]
            for path,title in specs:
                cores=read(35,path+'/matched_cluster_overlap.csv')
                cores['对应说明']=np.where(cores.intersection_three==0,'无三次共同成员','仅为成员匹配，需结合交并比判读')
                if not alignment_note_shown:
                    body+='<p>成员表以最大交集匹配统一编号，仅用于展示对应关系。无三次共同成员的行明确标记；稳定性结合实际规模、交集和交并比判断。以下各成员表采用同一口径。</p>'
                    alignment_note_shown=True
                body+=f'<h3>35维 · {title}：跨seed成员交集</h3>'+table(cores)+figure(path+'/pca_three_seeds.png',f'35维输入 · {title}；PCA仅展示，聚类在完整表示空间进行。')
        if exp=='Spectral':body+=figure('exp36_gps_distance_regression/spectral_comparison/all_settings_stability.png','35维输入：全部18组空间/参数设置的跨seed指标。')
        if exp=='Spectral':body+='<h3>相同谱聚类数值的来源核对</h3><p>以下两组在K=2、邻居5时，各seed逐户划分完全相同（按同簇家庭对核对，不受标签编号影响），因此汇总指标相同，未发现复制错误。</p>'+table(spectral_checks)
        if exp=='Exp32':body='<details><summary>早期探索结果（无置换null）</summary>'+body+'</details>'
        section(exp,exp+' · 34／35／36维结果',body)
    section('behavior','Exp37 · 行为预测重复核对','<p>Exp37回放始终使用完整36维策略，行为预测的输入为家庭特征与GPS。本次完整复跑后，以下表格在三种表示维度下逐项一致。</p>'+table(pd.DataFrame(invariance)))
    a=json.loads((BASE/'35d/exp38_raw_small_cluster_audit/audit.json').read_text())
    d=read(35,'exp38_raw_small_cluster_audit/all_households_data.csv')
    section('audit','Exp38 · 35维小簇家庭检查',f'<p>原始35维θ的小簇三次并集为{a["candidate_n"]}户，共同成员{a["core_n"]}户：'+html.escape('、'.join(a['core']))+'。</p>'+table(d[d.candidate][['household','minority_seed_count','city','zero_load_fraction','pv_peak_wh','complete_day_load_mean_kwh']])+f'<p>213个保存策略重新执行原家庭训练和验证回放，验证收益与日志最大误差为{a["val_replay_max_error"]:.2e} EUR/h。保存策略最大绝对参数为{a["max_abs_weight"]:.6f}，边界为±1.5。</p>')
    section('findings','GPS监督的实验观察','''
    <h3>同地点标签可被高度拟合，随机标签亦然</h3>
    <p>35维SupCon的同地点排序正确率为100%，标签置换后的平均值约99.988%。其Rain匹配簇在三个seed中的规模为19、21、21户，共同成员为Rain18户。这是样本内标签拟合；随机标签亦近乎完全拟合，与标签记忆相容。跨seed家庭对Jaccard为0.373，置换95%分位为0.451。地点标签拟合与家庭类型复现，在结果中是两个不同指标。</p>
    <h3>连续地理关系的留出检验</h3>
    <p>35维Exp34网络的训练地理排序正确率为81.13%，留出地点为56.98%，地图置换后的留出均值为57.21%（p=0.65）。35维Exp36留出到训练地点的距离MAE相对校准等权表示改善3.589 km，置换改善95%分位为4.003 km（p=0.13）；留出地点之间的改善为1.038 km（p=0.45）。地理排序未超过置换参照；距离回归未超过常数预测。</p>
    <h3>跨地点共同成员出现在局部配置</h3>
    <p>34、35、36维输入下，原始θ最小簇的三次共同成员均为10408与10495。35维SupCon除Rain核心外，另一个匹配簇仅共同保留6609；35维同地点三元组的主要匹配簇共同保留15户，全部位于Rain。36维同地点三元组的主要匹配簇共同保留29户，其中18户Rain、11户其他地点，另一个小簇无三次共同成员。</p>
    <p>同地点监督的重复核心主要集中在Rain。36维三元组主要簇有11户非Rain共同成员，未在34／35维三元组主要簇中重复。34维SupCon另有Bruchsal的10625与Wolfenbuttel的11361这2户共同成员；35维除Rain核心外仅另有6609一户。各配置未给出一致的跨地点成员组。</p>
    <p>35维SupCon的动作差距为−0.01968 kW，负荷形状差距为0.00914；同地点三元组分别为−0.03973 kW和0.03824。Exp37中，家庭特征加入经纬度后，主参数α=10的三个行为预测RMSE均上升。这些回放结果是策略描述，未提供独立的类型验证。当前结果未检测到可在上述设置间一致复现的地理相关策略类型。</p>''')
    section('local','局部正结果与多重比较',table(positive[['dimension','protocol','model','metric','observed','null_q95','valid_null']])+'''<p>上表列出Exp35中家庭对Jaccard、负荷差距或动作差距超过各自置换95%分位的全部组合。36维、保留Rain的三元组Jaccard为0.365（q95=0.305），负荷差距为0.0272（q95=0.0195），动作差距0.0192低于q95=0.0505。不含Rain时，其Jaccard为0.144，低于q95=0.159。该Jaccard增量未在34、35维复现。</p><p>局部负荷差距增量也见于35维三元组（0.0382，q95=0.0251），34维不含Rain的SupCon则在动作差距上超过参照。它们未形成跨协议、跨指标一致的结果，均作为探索性观察保留。</p><p>Exp35共列出3种维度×2种协议×2种监督模型×5个指标＝60个置换对照组合，指标彼此相关，部分观测或null汇总未定义；未实施多重比较校正。单独超过q95不等同于整体检验通过。Exp34/35各39次置换，全部39次有效时单侧置换p值的最小分辨率为1/40=0.025；有效null减少时分辨率更粗。</p>''')
    section('limits','数据与解释边界','''<p><strong>地理支持集中。</strong>重复地点仅4组，同坐标家庭对156对中153对来自Rain；不含Rain仅剩3对。监督正样本的构成限制了对跨地点共同类型的检验。</p><p><strong>窗口与测量。</strong>实际实验使用2026-04-15 02:00至04-27 00:00的287个小时记录，约12天。预处理脚本把15分钟记录汇总为小时电量。本报告分析分辨率为1小时。</p><p><strong>行为回放口径。</strong>action_gap与Exp37采用宿主完整287小时计算的局部缩放，并保留宿主地点输入s_idx。预处理脚本由经纬度边界匹配联邦州，再将州编码为s_idx。策略在不同宿主间同时面对缩放和地点输入变化；这种输入不是原家庭训练缩放。回放结果只能描述指定口径下的策略响应，不能作为θ簇的独立验证。</p><p><strong>参数重训与收益。</strong>此前34维参数空间的跨seed身份检索top-1为8.45%，采用另一seed的71户作为候选；该数值描述身份可辨识性，不是离散类型的判据。Exp37不使用改变后的嵌入，三种维度结果相同；Exp38按各输入维度重新核对小簇成员，三种维度仍得到相同的2户核心。收益采用Benefit-Cost口径，不直接解释为电费节省。</p><p><strong>维数敏感性。</strong>三种维度的地理排序、距离回归结论接近；部分聚类结果对输入坐标敏感。不含Rain的原始θ家庭对Jaccard依次为0.042、0.230、0.130。当前样本、表示和配置下，不能仅以簇数一致或一幅分离的PCA图判定稳定类型。</p>''')
    region_path=BASE/'region_supervision_exp39/summary.csv'
    if region_path.exists():
        SOURCES[str(region_path.relative_to(PACKAGE))]=hashlib.sha256(region_path.read_bytes()).hexdigest()
        region_results=pd.read_csv(region_path)
        primary=region_results.query("k==4 and protocol=='all' and model=='triplet_mlp' and metric=='balanced_accuracy'").iloc[0]
        raw_region=region_results.query("k==4 and protocol=='all' and model=='raw_centroid' and metric=='balanced_accuracy'").iloc[0]
        training_path=BASE/'region_supervision_exp39/full_panel_training_fit.csv'
        training_fit=pd.read_csv(training_path).query("k==4 and protocol=='all'")
        SOURCES[str(training_path.relative_to(PACKAGE))]=hashlib.sha256(training_path.read_bytes()).hexdigest()
        section('regions','Exp39：粗区域标签监督',f'<p>在51个唯一坐标的单位球三维坐标上构造K=3/4/5个区域。主配置预设K=4、35维、全部71户；正样本改为同区域、不同坐标家庭。采用四层64维三元组网络，5折留出坐标，199次坐标到区域标签置换。两种样本协议与所有K值完整报告。</p><p>主配置留出坐标balanced accuracy：网络{primary.observed:.2%}，原始θ同分类器对照{raw_region.observed:.2%}；网络置换均值{primary.null_mean:.2%}，95%分位{primary.null_q95:.2%}，单侧p={primary.p_upper:.3f}。六配置Holm校正p={primary.p_holm_six_configs:.3f}。K=4的均匀猜测机会水平为25%；网络、原始θ对照及置换均值均低于25%，本实验的推断参照是实际置换分布。全样本重代入balanced accuracy为{training_fit.resub_balanced_accuracy.min():.2%}–{training_fit.resub_balanced_accuracy.max():.2%}，留出为{primary.observed:.2%}，显示样本内拟合未推广到留出坐标。该任务检验θ能否预测粗地理区域，区域本身不是用电类型标签。</p>'+table(region_results.query("model=='triplet_mlp' and metric=='balanced_accuracy'"))+'<p><a href="../region_supervision_exp39/report.html">查看区域定义、全部结果、原始θ增量及聚类补充报告</a>。该实验采用四层网络与三元组损失，调整了样本构造和评估协议，不称为URL原实验的完整复现。</p>')
    evidence=pd.DataFrame([
        {'重点结果':'Exp34 · 35维网络 · 留出地理排序','观测':'56.98%','参照':'地图置换均值57.21%','检验':'p=0.65（39次置换）'},
        {'重点结果':'Exp36 · 35维 · 留出→训练地点MAE','观测':'117.655 km','参照':'常数113.340 km；等权121.244 km','检验':'相对等权改善p=0.13；与常数仅描述性比较'},
        {'重点结果':'Exp39 · K=4/all · 留出坐标BA','观测':'20.82%；训练重代入76.39%–86.38%','参照':'置换均值23.87%；均匀猜测25%','检验':'p=0.775（199次置换）'}])
    section('evidence','重点检验：观测与参照',table(evidence)+'<p>Exp39的K=4/all/35维为运行前固定的主配置。Exp34与Exp36在此作为报告重点结果展示，属于看过既有结果后的组织选择，不追溯声称为事前主检验；其余配置、聚类、局部正结果与行为关联均按探索性分析呈现。</p><p>Exp39使用L2逻辑回归基线、增加网络训练更新数的敏感性、Exp34增加到199次置换，以及Exp36增加常数项，均尚未运行。当前网络每次拟合256个三元组、10轮、批量32，共80步更新；区域监督的负结果限定于现有模型与训练配置，不能排除分类器或训练设置的影响。</p>')
    downloads=''.join('<li><a download="'+p.name+'" href="data:text/csv;base64,'+base64.b64encode(p.read_bytes()).decode()+'">'+p.name+'</a></li>' for p in sorted(out.glob('*.csv')) if not p.name.startswith('meter_quantization'))
    section('files','文件与复现','<p>代码：<code>code/gps/</code>；结果：<code>results/gps/34d/</code>、<code>35d/</code>、<code>36d/</code>；运行输入：<code>data/gps/</code>。历史运行清单保留当时路径，迁移映射见<code>results/gps/migration_manifest.json</code>。</p><p>共同场景回放沿用宿主全窗缩放和地点输入，各策略独立演化SOC；Exp38使用原家庭训练段和验证段各自的缩放。收益使用既有Benefit-Cost口径。动作差距是簇间平均动作距离减簇内平均动作距离，单位kW；负荷差距使用归一化24小时负荷曲线。</p><ul>'+downloads+'</ul>')
    css='''*{box-sizing:border-box}body{margin:0;background:#f4f6f9;color:#203148;font:15px/1.8 -apple-system,BlinkMacSystemFont,"PingFang SC",sans-serif}header{background:#17334f;color:white;padding:44px max(24px,calc((100vw - 1150px)/2))}h1{font-size:32px;line-height:1.4}main{max-width:1250px;margin:auto;padding:20px}nav{background:white;padding:12px 20px;position:sticky;top:0;z-index:2;white-space:nowrap;overflow:auto}nav a{margin-right:22px;color:#24517c;text-decoration:none}section{background:white;border:1px solid #dce3ec;border-radius:8px;padding:28px;margin:20px 0;scroll-margin-top:60px}h2{color:#24517c;font-size:23px}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:12px}th,td{text-align:left;padding:9px;border-bottom:1px solid #dce3ec;white-space:nowrap}th{background:#eaf0f7}tr:nth-child(even){background:#f8fafc}figure{margin:24px 0}img{width:100%;height:auto}figcaption{font-size:13px;color:#52677d}code{overflow-wrap:anywhere}@media(max-width:600px){main{padding:8px}section{padding:16px}h1{font-size:26px}}@media print{nav{display:none}figure{break-inside:avoid}}'''
    css+='''.pipeline{display:flex;align-items:center;justify-content:center;gap:10px;flex-wrap:wrap;background:#eef3f9;padding:20px;border-radius:6px;margin:24px 0}.pipeline span{padding:8px 12px;background:white;border:1px solid #cbd8e6;border-radius:4px}.signal{text-align:center;color:#24517c;padding:10px;border-bottom:2px solid #cbd8e6}h3{margin-top:30px;color:#2b4a69}'''
    names={'regions':'区域标签补充','summary':'摘要','limits':'数据与解释边界','local':'局部正结果','guidance':'研究问题','mapping':'GPS如何监督','evaluation':'评价流程','overview':'维度对照','behavior':'行为预测','audit':'小簇检查','findings':'实验观察','files':'文件索引'}
    main_keys=['summary','evidence','regions','limits']
    ordered=[item for key in main_keys for item in sections if item[0]==key]
    appendix=[item for item in sections if item[0] not in main_keys and item[0]!='files']
    appendix_html='<section id="appendix"><h2>附录：方法、全量结果与相关核对</h2>'+''.join('<details><summary>'+html.escape(title)+'</summary>'+body+'</details>' for key,title,body in appendix)+'</section>'
    ordered.append(('appendix','附录',appendix_html));ordered.extend(item for item in sections if item[0]=='files')
    sections=ordered
    names.update({'evidence':'重点检验','regions':'区域监督','appendix':'附录'})
    css+='details{border-top:1px solid #dce3ec;padding:16px 0}summary{cursor:pointer;font-weight:600;color:#24517c}details section{border:0;padding:12px 0;margin:0} @media print{details{display:block}}'
    page='<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>GPS 34／35／36维实验对照</title><style>'+css+'</style></head><body><header><h1>GPS 表示学习<br>34／35／36维完整实验对照</h1><p>地理监督设计、映射方法、成员复现与策略行为</p><p>71户 · CEM seed 0／1／42 · 实验汇总2026-11-08 · 修订2026-11-09</p></header><nav>'+''.join(f'<a href="#{k}">{html.escape(names.get(k,k))}</a>' for k,_,_ in sections)+'</nav><main>'+''.join(b for _,_,b in sections)+'</main><script>window.addEventListener("beforeprint",()=>document.querySelectorAll("details").forEach(e=>{e.dataset.previous=e.open?"1":"0";e.open=true}));window.addEventListener("afterprint",()=>document.querySelectorAll("details").forEach(e=>e.open=e.dataset.previous==="1"));</script></body></html>'
    report=out/'gps_dimensions_report_2026-11-08.html';report.write_text(page)
    (out/'report_sources.json').write_text(json.dumps({'sources':SOURCES,'builder_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'report_sha256':hashlib.sha256(report.read_bytes()).hexdigest()},indent=2))
    print('Saved',report)


if __name__=='__main__':main()
