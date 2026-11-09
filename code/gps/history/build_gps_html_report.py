"""Build a self-contained factual HTML report from saved GPS experiment outputs."""
from pathlib import Path
import base64
import hashlib
import html
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "04.results/shape"
OUT = ROOT / "document/gps_experiment_report_2026-11-07_v2.html"
SEEDS = ["cem_seed0", "cem_seed1", "cem_seed42"]
SOURCES = {}


def read(rel):
    path = RESULTS / rel
    SOURCES[rel] = path
    return pd.read_csv(path)


def esc(value):
    return html.escape(str(value))


def num(value, digits=3):
    return "—" if pd.isna(value) else f"{float(value):.{digits}f}"


def pct(value, digits=1):
    return "—" if pd.isna(value) else f"{100*float(value):.{digits}f}%"


def table(headers, rows, caption=""):
    return '<div class="table-wrap"><table>' + (f'<caption>{esc(caption)}</caption>' if caption else '') + '<thead><tr>' + ''.join(f'<th>{esc(x)}</th>' for x in headers) + '</tr></thead><tbody>' + ''.join('<tr>'+''.join(f'<td>{esc(x)}</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>'


def figure(rel, caption, compact=False):
    path = RESULTS / rel
    image = base64.b64encode(path.read_bytes()).decode()
    return f'<figure class="{"compact" if compact else ""}"><a href="data:image/png;base64,{image}" target="_blank" aria-label="打开原图"><img src="data:image/png;base64,{image}" alt="{esc(caption)}" loading="lazy"></a><figcaption>{esc(caption)}</figcaption></figure>'


def section(identifier, number, title, body):
    return f'<section id="{identifier}"><div class="section-label">{number}</div><h2>{title}</h2>{body}</section>'


def summary_rows(frame, protocol="all"):
    d = frame[frame.protocol == protocol]
    rows = []
    for m, label in [("raw", "原始 θ"), ("supcon", "SupCon"), ("triplet", "同地点三元组网络")]:
        f = d[d.model == m].set_index("metric")
        rows.append([label, pct(f.loc["place_fit", "observed"], 3), num(f.loc["jaccard", "observed"]),
                     num(f.loc["jaccard", "null_mean"]), num(f.loc["action_gap_kw", "observed"], 5), num(f.loc["load_gap", "observed"], 5)])
    return rows


def main():
    a32 = read("exp32_gps_triplet/summary.csv")
    a33 = read("exp33_gps_cluster_comparison/comparison.csv")
    cov33 = read("exp33_gps_cluster_comparison/cluster_coverage.csv")
    a34 = read("exp34_gps_network_discovery/observed_by_seed.csv")
    sum34 = read("exp34_gps_network_discovery/summary.csv")
    met35 = read("exp35_place_networks/cluster_metrics.csv")
    sum35 = read("exp35_place_networks/summary.csv")
    core35 = read("exp35_place_networks/cluster_visualization/cluster_overlap_summary.csv")
    cv36 = read("exp36_gps_distance_regression/cv_summary.csv")
    perm36 = read("exp36_gps_distance_regression/permutation_comparison.csv")
    labels36 = read("exp36_gps_distance_regression/labels.csv")
    core36 = read("exp36_gps_distance_regression/cluster_visualization/weighted/matched_cluster_overlap.csv")
    raw36 = read("exp36_gps_distance_regression/cluster_visualization/weighted/small_cluster_vs_raw.csv")
    spectral = "exp36_gps_distance_regression/spectral_comparison/"
    sp = read(spectral + "cross_seed_summary.csv")
    spcore = read(spectral + "cluster_core.csv")
    spcmp = read(spectral + "raw_vs_exp36.csv")
    graph = read(spectral + "graph_diagnostics.csv")
    a37 = read("exp37_gps_behavior_increment/summary.csv")
    stability37 = read("exp37_gps_behavior_increment/stability.csv")
    audit_path = "exp38_raw_small_cluster_audit/"
    data38 = read(audit_path + "all_households_data.csv")
    params38 = read(audit_path + "all_households_parameters_training.csv")
    overlap38 = read(audit_path + "small_cluster_overlap.csv")
    candidate38 = read(audit_path + "candidate_10_audit.csv")
    retrieval38 = read(audit_path + "cross_seed_retrieval.csv")
    core38 = params38[params38.core]
    reference38 = data38[~data38.candidate]
    raw_members = []
    for s in SEEDS:
        d = params38[params38.seed == s]
        m = d[d.in_small_cluster]
        raw_members.append([s.replace('cem_', ''), len(m), '、'.join(m.household.astype(str)), int((d.cluster == -1).sum())])
    primary = pd.read_excel(ROOT / "00.data/raw/家庭经纬度信息.xlsx", dtype={"id": str})
    copy = pd.read_excel(ROOT / "00.data/raw/副本家庭经纬度信息.xlsx", sheet_name="教授团队", dtype={"id": str})
    assert primary.sort_values("id").reset_index(drop=True).equals(copy.sort_values("id").reset_index(drop=True))
    valid = primary.dropna(subset=["longitude", "latitude"])
    locations = valid.groupby(["country", "city"])
    assert locations.apply(lambda x: len(x[["longitude", "latitude"]].drop_duplicates()), include_groups=False).max() == 1
    ids = sorted(p.name.removesuffix("_expert.npy") for p in (ROOT / "04.results/cem_npy/seed_wise/cem_seed42").glob("*_expert.npy"))
    training = primary[primary.id.isin(ids)]
    assert len(training) == 71
    place_counts = training.groupby(["city", "longitude", "latitude"]).size().sort_values(ascending=False)
    assert len(place_counts) == 51 and place_counts.max() == 18

    blocks = []
    blocks.append(section("data", "01", "数据与坐标核对", f'''
    <p>输入为 71 户德国家庭的 CEM 策略向量，每户包含 seed 0、1、42 三次训练结果，共 213 条向量。原始 θ 为 36 维；表示学习使用排除 location 权重与 bias 后的 34 维坐标。每个 seed 分别进行标准化和映射拟合。</p>
    <div class="facts"><div><strong>71</strong><span>家庭</span></div><div><strong>51</strong><span>不同坐标组</span></div><div><strong>3</strong><span>CEM seed</span></div><div><strong>34</strong><span>输入维数</span></div></div>
    <p>原始坐标表含 {len(primary):,} 户，{len(valid):,} 户具备经纬度，11 户经纬度为空。有效记录涉及 {len(locations)} 个“国家＋地名”组合；每个组合均只有一对坐标。原表与副本按家庭 ID 对齐后的记录一致。训练集 71 户的坐标与原表逐户一致。</p>
    {table(['地点','原表户数','训练户数','经度','纬度'], [[city, int(sum((primary.country=='Germany')&(primary.city==city))), int(sum(training.city==city)), num(training[training.city==city].longitude.iloc[0],6), num(training[training.city==city].latitude.iloc[0],6)] for city in ['Rain','Heilbronn','Berlin','Maselheim']])}
    <p>训练集中，Rain 为 18 户，Heilbronn、Berlin、Maselheim 各 2 户，另外 47 个坐标组各 1 户。共有 156 对同坐标家庭，其中 Rain 提供 153 对。主分析保留全部 71 户。</p>
    <p class="note">时序窗口：2026-04-15 02:00 至 2026-04-27 00:00，共 287 个小时记录。地理距离由经纬度经 Haversine 公式计算，单位为公里。</p>'''))

    overview = [
        ["Exp32", "地理远近排序", "非负对角度量，34维", "逐地点留出；地理排序与近邻动作距离"],
        ["Exp33", "同地点／地理排序／二者混合", "非负对角度量，34维", "HDBSCAN；标签置换；跨 seed 一致性"],
        ["Exp34", "地理远近排序", "对角度量34维；三元组网络64维", "5折地点分组；HDBSCAN；地图置换"],
        ["Exp35", "同坐标地点标签", "SupCon 8维；同地点三元组网络64维", "HDBSCAN；地点标签置换；动作和负荷差异"],
        ["Exp36", "连续球面距离", "非负对角距离回归，34维", "5折地点分组；HDBSCAN；99次地图置换"],
        ["Exp36 谱聚类对照", "沿用已保存表示", "原始标准化 θ 与 Exp36 z", "54次谱聚类；成员交集与邻居数敏感性"],
        ["Exp37", "纬度、经度作为输入特征", "家庭特征／家庭特征＋GPS 岭回归", "51次地点留出；策略行为预测误差"],
        ["Exp38", "原始θ小簇的补充检查", "既有36维策略；34维几何坐标", "成员复现；参数边界；输入数据；训练日志与验证收益复现"],
    ]
    blocks.append(section("methods", "02", "实验流程与评价指标", '''
    <div class="pipeline"><span>θ 的34维坐标</span><b>→</b><span>标准化</span><b>→</b><span>GPS 监督映射 f</span><b>→</b><span>新表示 z</span><b>→</b><span>聚类及成员核对</span></div>
    ''' + table(["实验", "GPS 使用方式", "模型／表示", "评估内容"], overview) + '''
    <p>HDBSCAN 统一使用 <code>min_cluster_size=4</code>、<code>min_samples=2</code>；未归簇样本记为 −1。谱聚类采用近邻图，输入簇数 K∈{2,3,4}，邻居参数∈{5,10,15}，标签分配使用 k-means，随机种子42、n_init=10。</p>
    <dl><dt>地理排序正确率</dt><dd>以一个地点为参照，比较表示空间与地理空间对另外两个地点的远近排序；平局计0.5。</dd>
    <dt>同地点排序正确率</dt><dd>比较同地点家庭对与异地点家庭对的表示距离，先在地点内计算，再对有重复家庭的地点等权平均。</dd>
    <dt>家庭对 Jaccard</dt><dd>两个 seed 中共同归为一簇的家庭对的交集除以并集。HDBSCAN 的未归簇点不构成同簇家庭对。</dd>
    <dt>逐簇三次交并比</dt><dd>以seed42为参照，按最大成员交集对齐簇编号，再计算三个seed的成员交集／并集。</dd>
    <dt>动作差距 Δaction</dt><dd>簇间家庭对平均动作距离减去簇内家庭对平均动作距离，单位kW；距离由各策略在共同外生场景中的72小时回放计算，各策略拥有自己的SOC轨迹。</dd>
    <dt>负荷差距 Δload</dt><dd>归一化24小时平均负荷曲线之间的距离，取簇间平均值减去簇内平均值。</dd></dl>
    <h3>行为回放的输入口径</h3>
    <p>Exp34、Exp35、Exp37的共同场景回放使用宿主家庭完整287小时数据计算的局部缩放，以及宿主家庭的地点输入；回放窗口为末段72小时。策略保留完整36维参数，各策略从电池SOC上下限的中点出发，独立更新SOC。Exp35的负荷形状由完整287小时记录按小时分组、求平均后归一化得到。</p>
    <p>Exp38执行各策略在原家庭上的检查：训练段为前191小时，验证段为最后96小时，分别计算各段缩放；验证执行95步。训练检查使用起点0、59、118的三个72小时窗口。所列收益使用现有Benefit-Cost计算口径。</p>'''))

    rows32 = []
    for metric, label in [("theta","原始 θ"),("gps_triplet","GPS 对角度量")]:
        g=a32[(a32.protocol=="all")&(a32.metric==metric)]
        rows32.append([label, *[pct(g[g.seed==s].geographic_order_accuracy.iloc[0],2) for s in SEEDS], pct(g.geographic_order_accuracy.mean(),2), num(g.neighbor_action_gap_kw.mean(),4)])
    rows33=[]
    for arm,label in [("raw","原始 θ"),("place_id","同地点三元组"),("distance","地理排序三元组"),("mixed","各半混合")]:
        r=a33[(a33.protocol=="all")&(a33.arm==arm)].iloc[0]
        c=cov33[(cov33.protocol=="all")&(cov33.permutation==0)&(cov33.arm==arm)]
        rows33.append([label,'／'.join(str(int(c[c.seed==s].n_clusters.iloc[0])) for s in SEEDS), num(r.observed_jaccard), num(r.null_mean), num(r.null_q95), int(r.valid_null)])
    blocks.append(section("exp32-33","03","Exp32–33：对角度量与监督形式对照",'''
    <p>Exp32 使用地理排序三元组学习各维非负权重。按坐标组逐一留出，比较地理排序正确率及最近5户的平均动作距离。</p>'''+table(["表示","seed 0","seed 1","seed 42","平均排序正确率","近邻动作距离均值（kW）"],rows32)+'''
    <p>Exp33 对照同地点、地理距离排序及各半混合三种三元组来源。每个分支使用1,024个三元组和相同的正则化对角度量目标；实施99次家庭与地点标签对应关系置换。</p>'''+table(["监督分支","簇数 0／1／42","真实家庭对J","置换均值","置换95%分位","有效置换数"],rows33)+'''
    <p class="note">有效置换数要求三个seed配对的一致性均有定义；原始θ分支未实施标签置换。</p>'''))

    rows34=[]
    raw_members_block = '<h3>原始θ基准的小簇成员</h3>' + table(
        ['seed','小簇户数','成员ID','未归簇户数'], raw_members) + table(
        ['seed配对','小簇交集','小簇并集','小簇Jaccard'],
        [[r.seed_a.replace('cem_','')+'／'+r.seed_b.replace('cem_',''), int(r.intersection), int(r.union), num(r.jaccard)] for r in overlap38.itertuples()]) + '''
        <p>小簇三次并集为10户，三次共同成员为10408、10495，三次交并比为0.200。主体簇规模依次为65、66、64户。原始θ的家庭对Jaccard约0.816，对应全部已归簇家庭对的汇总；上表单列小簇成员的交并比。Exp38对这10户及其余61户进行参数、数据与训练记录检查。</p>'''
    for model,label in [("raw","原始 θ"),("diagonal","GPS 对角度量"),("network","GPS 三元组网络")]:
        g=a34[a34.model==model]
        v=sum34[(sum34.model==model)&(sum34.metric=="test")].iloc[0]
        j=sum34[(sum34.model==model)&(sum34.metric=="jaccard")].iloc[0]
        rows34.append([label,pct(g.train.mean(),2),pct(g.test.mean(),2),pct(v.null_mean,2),num(v.p_upper,3),'／'.join(str(int(g[g.seed==s].k.iloc[0])) for s in SEEDS),num(j.observed)])
    blocks.append(section("exp34","04","Exp34：地理排序监督的两种映射",'''
    <p>对角分支对标准化θ逐维加权。网络分支采用 <code>34→64→64→64→64</code>，前三层使用ReLU，输出为64维；使用欧氏三元组间隔损失，间隔1、学习率0.01，训练10轮，每轮32个批次，每次训练使用1,024个三元组。网络初始化种子固定为34。</p>
    <p>评估采用5折坐标组划分，并实施39次地点坐标置换。排序正确率先按地点汇总，再对地点与三个CEM seed取平均。聚类使用全样本拟合的表示。</p>'''+table(["表示","训练排序","留出排序","置换留出均值","置换p值","HDBSCAN簇数 0／1／42","跨seed家庭对J"],rows34)+'''
    <p>网络分支的训练排序正确率为79.96%，留出地点为55.88%；地图置换后的留出均值为56.91%。三个seed的网络表示分别得到0、3、2簇。</p>'''+raw_members_block+figure("exp34_gps_network_discovery/cluster_visualization/diagonal/pca_three_seeds.png","图1．Exp34 对角度量的全样本表示与 HDBSCAN 划分。灰色为未归簇点，黑圈为Rain。")+figure("exp34_gps_network_discovery/cluster_visualization/network/pca_three_seeds.png","图2．Exp34 三元组网络表示。seed0有71户未归簇；seed1和seed42分别有36户、31户未归簇。")))

    obs35=met35[(met35.protocol=="all")&(met35.rep==0)]
    rows35=[]
    for model,label in [("supcon","SupCon"),("triplet","同地点三元组网络")]:
        for s in SEEDS:
            r=obs35[(obs35.model==model)&(obs35.seed==s)].iloc[0]
            rows35.append([label,s.replace("cem_",""),int(r.k),int(round(r.coverage*71)),71-int(round(r.coverage*71)),pct(r.place_fit,3)])
    c35=[[r.cluster,int(r.cem_seed0),int(r.cem_seed1),int(r.cem_seed42),int(r.intersection_three),int(r.union_three),num(r.jaccard_three),int(r.rain_in_core)] for r in core35.itertuples()]
    blocks.append(section("exp35","05","Exp35：同地点监督对比学习",'''
    <p>相同经纬度定义为相同地点。同地点、不同家庭构成正样本对，其他地点家庭进入对比项。SupCon网络结构为 <code>34→32→8</code>，隐藏层使用ReLU，8维输出进行单位长度归一化。温度0.1、学习率0.03，200次全批量更新，初始化种子35。正样本锚点按地点规模加权。</p>
    <p>同地点三元组分支使用相同地点的两户与另一地点的一户构造三元组，沿用Exp34的64维网络。两个分支均实施39次地点标签置换。</p>'''+table(["分支","CEM seed","簇数","已归簇户数","未归簇户数","同地点排序正确率"],rows35)+'''
    <h3>SupCon 的四簇成员核对</h3>'''+table(["对齐簇","seed 0","seed 1","seed 42","三次交集","三次并集","三次交并比","交集中Rain户数"],c35)+'''
    <p>三个seed均输出4簇。C2的三次共同成员为Rain的18户；seed1中该簇另外包含11户。C4的三次共同成员为10625与11361。C1、C3的三次共同成员数为0。</p>'''+figure("exp35_place_networks/cluster_visualization/pca_three_seeds.png","图3．Exp35 SupCon 的8维表示投影与四簇划分。颜色按成员重合与seed42对齐。")+'''
    <h3>标签拟合、跨seed一致性与行为差距</h3>'''+table(["分支","同地点排序","真实家庭对J","置换家庭对J均值","Δaction（kW）","Δload"],summary_rows(sum35))+'''
    <p>SupCon真实标签下的同地点排序正确率为99.996%，置换标签下均值为99.969%；家庭对Jaccard分别为0.351与0.358。全样本SupCon的Δaction为−0.00458 kW，Δload为0.01272。</p>
    <details><summary>逐户查看三个seed的簇归属</summary>'''+figure("exp35_place_networks/cluster_visualization/household_membership_three_seeds.png","Exp35 SupCon：每行同一家庭，每列一个CEM seed。")+'</details>'))

    distance_rows=[]
    for m,label in [("constant","常数距离预测"),("raw_scaled","原始等权距离＋尺度校准"),("weighted","GPS 加权距离")]:
        d=cv36[(cv36.rep==0)&(cv36.model==m)].set_index("split")
        distance_rows.append([label,num(d.loc['train','mae_km'],2),num(d.loc['test_to_train','mae_km'],2),num(d.loc['test_to_test','mae_km'],2)])
    c36=[["主体组" if r.aligned_cluster=="C1" else "少数组",int(r.cem_seed0),int(r.cem_seed1),int(r.cem_seed42),int(r.intersection_three),int(r.union_three),num(r.jaccard_three)] for r in core36.itertuples()]
    blocks.append(section("exp36","06","Exp36：连续地理距离回归",'''
    <p>以球面地理距离除以训练地点对的距离中位数作为连续监督目标，拟合非负维度权重。训练只使用不同坐标的家庭对，各地点对等权。权重通过L-BFGS-B优化，正则系数0.1，向等权距离的校准系数收缩。</p>
    <div class="equation">z<sub>ik</sub> = x<sub>ik</sub> √(w<sub>k</sub> / 34)；&nbsp; d̂<sub>ij</sub> = s · ‖z<sub>i</sub> − z<sub>j</sub>‖<sub>2</sub></div>
    <p class="note">x为标准化后的θ34，w为非负权重，s为训练集地理距离尺度。该分支没有神经网络和激活函数。</p>
    <p>采用5折坐标组划分、99次地点坐标置换。误差先按地点对平衡，再对各折及三个seed等权平均。</p>'''+table(["模型","训练MAE（km）","留出地点→训练地点 MAE（km）","留出地点之间 MAE（km）"],distance_rows)+table(["评估家庭对","相对等权距离的MAE改善（km）","置换改善均值（km）","置换95%分位（km）","单侧p值"],[["留出→训练" if r.split=="test_to_train" else "留出→留出",num(r.mae_improvement_km,3),num(r.null_mean,3),num(r.null_q95,3),num(r.p_upper,2)] for r in perm36.itertuples()])+'''
    <h3>HDBSCAN 成员结果</h3>'''+table(["对齐组","seed 0","seed 1","seed 42","三次交集","三次并集","三次交并比"],c36)+'''
    <p>未归簇户数依次为6、4、5。主体组的三次共同成员为51户；少数组的三次共同成员为10495，三次并集共10户。家庭对Jaccard三对seed平均为0.685。</p>'''+table(["seed","原始空间小簇","Exp36小簇","两空间交集","两空间并集","交并比"],[[r.seed.replace("cem_",""),r.raw_small_n,r.weighted_small_n,r.intersection,r.union,num(r.jaccard)] for r in raw36.itertuples()])+figure("exp36_gps_distance_regression/cluster_visualization/weighted/pca_three_seeds.png","图4．Exp36 连续地理距离回归的34维表示投影及 HDBSCAN 两簇划分。")))

    small=spcore[(spcore.k==2)&(spcore.aligned_cluster==2)]
    sp_rows=[["原始 θ" if r.model=="raw_theta" else "Exp36",int(r.n_neighbors),int(r.cem_seed0),int(r.cem_seed1),int(r.cem_seed42),int(r.intersection_three),int(r.union_three),num(r.jaccard_three)] for r in small.itertuples()]
    blocks.append(section("spectral","07","Exp36 补充：谱聚类对照",'''
    <p>固定原始标准化θ与Exp36已保存表示，使用同一谱聚类参数网格：2个空间 × 3个CEM seed × 3个K × 3个邻居数，共54次拟合、3,834条家庭归属记录。谱聚类在完整34维输入上构建近邻图。K为输入参数。</p>
    <h3>K=2 时的少数组成员</h3>'''+table(["空间","邻居参数","seed 0","seed 1","seed 42","三次交集","三次并集","三次交并比"],sp_rows)+'''
    <p>邻居参数10时，原始θ的少数组分别为9、8、8户，三次交集为0户；Exp36分别为25、7、7户，三次共同成员为10408、10495，均属Rain。</p>'''+figure(spectral+"k2_neighbors10_pca.png","图5．谱聚类，指定K=2、邻居参数10。上排为原始θ，下排为Exp36；每列对应一个CEM seed。")+'''
    <h3>全部参数组合的跨seed指标</h3>'''+figure(spectral+"all_settings_stability.png","图6．全部18个空间／参数设置的稳定性指标。左：跨seed平均ARI；中：家庭对Jaccard；右：各匹配簇三次交并比的最小值。")+'''
    <p>Exp36在9组参数下的跨seed平均ARI为0.010–0.202；原始θ为0.015–0.322。18张近邻图中17张连通；原始θ、seed42、邻居参数5的图含2个连通分量。Exp36的9张图均连通。</p>
    <h3>同一个seed中，原始θ与Exp36的划分对应</h3>'''+table(["seed","ARI","家庭对Jaccard","匹配后相同成员比例"],[[r.seed.replace('cem_',''),num(r.ari),num(r.pair_jaccard),pct(r.matched_fraction,1)] for r in spcmp[(spcmp.k==2)&(spcmp.n_neighbors==10)].itertuples()])+'''
    <p class="note">表中设置为K=2、邻居参数10。原始θ seed42的9组参数结果与此前保存的谱聚类结果逐组比较，ARI均为1。</p>
    <details><summary>全部18组参数的数值结果</summary>'''+table(["空间","邻居参数","指定K","跨seed平均ARI","家庭对Jaccard","匹配成员比例"],[["原始 θ" if r.model=="raw_theta" else "Exp36",r.n_neighbors,r.k,num(r.ari),num(r.pair_jaccard),pct(r.matched_fraction)] for r in sp.itertuples()])+'</details>'))

    names37={"charge_kwh_per_day":"日均充电量（kWh）","charge_share_10_16":"10–16点充电占比","soc_lower_fraction":"SOC下限时步比例"}
    blocks.append(section("exp37","08","Exp37：GPS 对策略行为预测的增量",'''
    <p>输入家庭特征为前191小时的平均负荷、平均PV（均取log1p）、零负荷比例、负荷变异系数、平均云量及平均短波辐照度。GPS扩展模型增加纬度和经度两列。采用岭回归，主参数α=10，另计算α=1、100。</p>
    <p>目标为每条策略在共同71个家庭末段72小时外生场景中回放后的平均行为：日均充电量、10:00至16:00前的充电占比，以及动作前SOC处于下限的时步比例。按51个坐标组逐一留出；误差按地点及seed等权汇总。</p>'''+table(["行为指标","均值对照RMSE","家庭特征RMSE","家庭特征＋GPS RMSE"],[[names37[r.metric],num(r.constant_rmse,5),num(r.profiles_rmse,5),num(r.gps_rmse,5)] for r in a37[a37.alpha==10].itertuples()])+'''
    <p>α=10时，三个指标加入GPS后的总体RMSE均上升。α=1、100时三个指标的总体变化方向相同。</p>'''+table(["行为指标","seed0–1 排序相关","seed0–42 排序相关","seed1–42 排序相关"],[[names37[m],*[num(v) for v in stability37[stability37.metric==m].rho]] for m in names37])+'''
    <details><summary>三个正则参数的全部RMSE结果</summary>'''+table(["α","指标","均值对照","家庭特征","家庭特征＋GPS"],[[num(r.alpha,0),names37[r.metric],num(r.constant_rmse,5),num(r.profiles_rmse,5),num(r.gps_rmse,5)] for r in a37.itertuples()])+'</details>'))

    audit_rows = []
    for r in data38[data38.candidate].itertuples():
        d = params38[params38.household == r.household].set_index('seed')
        audit_rows.append([str(r.household), int(r.minority_seed_count), r.city,
                           pct(r.zero_load_fraction,2), num(r.pv_peak_wh,0),
                           num(r.complete_day_load_mean_kwh,3),
                           '／'.join(str(int(d.loc[s,'best_iter'])) for s in SEEDS)])
    distribution_rows = []
    for label, d in [('核心2户', data38[data38.core]), ('其余候选8户', data38[data38.candidate & ~data38.core]), ('参考61户', reference38)]:
        distribution_rows.append([label, len(d), pct(d.zero_load_fraction.median(),2),
                                  num(d.pv_peak_wh.median(),0), num(d.complete_day_load_mean_kwh.median(),3)])
    blocks.append(section('exp38','09','Exp38：原始θ小簇候选家庭检查',f'''
    <p>以原始θ的HDBSCAN小簇三次并集10户为候选组、三次交集10408与10495为核心2户，并以候选并集之外的61户为参考组。共检查71户、213个已保存策略；重新计算的原始θ簇标签与Exp34保存结果逐项一致。</p>
    <h3>候选家庭的数据与训练记录</h3>''' + table(
        ['家庭ID','进入小簇次数','地点','零负荷比例','PV峰值（Wh）','完整日平均负荷（kWh/日）','最佳迭代 0／1／42'], audit_rows) + '''
    <p class="note">数据取自实际训练使用的预处理CSV。PV峰值指单个小时记录的最大电量；日负荷只使用完整24小时日计算。最佳迭代为验证收益首次达到最大值的迭代编号，从0起算。</p>
    <h3>三组家庭的描述性对照</h3>''' + table(
        ['组别','户数','零负荷比例中位数','PV峰值中位数（Wh）','日负荷中位数（kWh/日）'],distribution_rows) + f'''
    <p>候选10户的零负荷比例全部低于参考61户的中位数{pct(reference38.zero_load_fraction.median(),2)}。核心户10408、10495均位于Rain，其PV峰值为900、800 Wh；参考组PV峰值范围为{num(reference38.pv_peak_wh.min(),0)}–{num(reference38.pv_peak_wh.max(),0)} Wh。核心两户在各seed的标准化质心距离处于参考61户的{num(core38.standardized_centroid_distance_pct_reference61.min(),2)}–{num(core38.standardized_centroid_distance_pct_reference61.max(),2)}百分位。</p>
    <h3>参数与训练记录核对</h3>
    <p>全部213个保存策略在±1.5边界处的参数数量为0（判定容差10⁻⁶），最大绝对参数为{num(params38.max_abs_weight.max(),6)}。检查对象为最终保存的CEM分布中心。核心两户的最佳迭代分布在{int(core38.best_iter.min())}–{int(core38.best_iter.max())}代。</p>
    <p>全部213个策略在各自家庭验证环境中的回放收益与日志最高验证收益一致，最大绝对误差为{params38.val_replay_abs_error.max():.2e} EUR/h。seed42有71户保存了逐代参数轨迹，其最佳迭代参数与保存策略逐项相同。其余两个seed提供训练收益历史，未保存逐代参数轨迹。</p>
    <p>71户预处理负荷/PV列的缺失值、负值、重复时间戳及非小时间隔数量均为0。</p>''' + figure(
        audit_path+'audit_distributions.png','图7．核心2户、其余候选8户与参考61户的输入数据和参数分布。下排合并三个seed作描述性展示。') + figure(
        audit_path+'core_training_curves.png','图8．核心两户的验证收益与候选种群训练均值；灰色为参考61户的中位数及四分位区间。') + '''
    <details><summary>核心两户的参数与checkpoint回放收益</summary>''' + table(
        ['家庭','seed','权重范数（不含偏置）','偏置','训练固定三窗口（EUR/h）','验证95步（EUR/h）'],
        [[str(r.household),r.seed.replace('cem_',''),num(r.w_norm,4),num(r.bias,4),num(r.replay_checkpoint_train_fixed3_eur_h,6),num(r.replay_checkpoint_val_eur_h,6)] for r in core38.itertuples()]) + '''
    <p>训练列评估保存策略在三个固定窗口中的平均收益；学习曲线中的mean_reward记录该代候选种群在随机训练窗口上的平均收益。</p></details>
    <details><summary>补充：跨seed家庭身份检索</summary>''' + f'''
    <p>使用去除地点与偏置后的34维参数，在全部213条记录上拟合共同标准化尺度。每条查询在另一个seed的71户中检索，共6个有方向的seed配对。平均top-1为{pct(retrieval38.top1.mean(),2)}，top-5为{pct(retrieval38.top5.mean(),2)}；等概率参考分别为{pct(1/71,2)}和{pct(5/71,2)}。该项为家庭身份检索描述，六个配对重复使用同一批71户。</p></details>'''))

    sensitivity=[]
    for model,label in [("raw","原始 θ"),("supcon","SupCon"),("triplet","同地点三元组网络")]:
        d=met35[(met35.protocol=="exclude_largest")&(met35.rep==0)&(met35.model==model)]
        f=sum35[(sum35.protocol=="exclude_largest")&(sum35.model==model)].set_index("metric")
        sensitivity.append([label,'／'.join(str(int(d[d.seed==s].k.iloc[0])) for s in SEEDS),num(f.loc['jaccard','observed']),num(f.loc['jaccard','null_q95']),num(f.loc['action_gap_kw','observed'],5),num(f.loc['action_gap_kw','null_q95'],5)])
    blocks.append(section("appendix","10","补充结果与计算口径",'''
    <details><summary>Exp35：53户子集（不含Rain）的已保存结果</summary>
    <p>该子集包含53户、50个坐标组，3个坐标组各2户，同地点家庭配对共3对。</p>'''+table(["分支","簇数 0／1／42","家庭对J","置换J 95%分位","Δaction（kW）","置换Δaction 95%分位"],sensitivity)+'''<p class="note">表中的“—”对应三个seed汇总时存在未定义结果。SupCon的Δaction为0.10585 kW，家庭对Jaccard为0.185。</p></details>
    <details><summary>Exp32–33：53户子集的已保存结果</summary>'''+table(["Exp32表示","地理排序正确率（三seed均值）","近邻动作距离（kW）"],[["原始 θ" if m=="theta" else "GPS 对角度量",pct(g.geographic_order_accuracy.mean(),2),num(g.neighbor_action_gap_kw.mean(),4)] for m,g in a32[a32.protocol=='exclude_largest'].groupby('metric')])+table(["Exp33分支","家庭对J","置换均值","置换95%分位","有效置换数"],[[r.arm,num(r.observed_jaccard),num(r.null_mean),num(r.null_q95),int(r.valid_null)] for r in a33[a33.protocol=='exclude_largest'].itertuples()])+'''</details>
    <p>置换方式：Exp33、Exp35改变家庭与地点标签的对应关系；Exp34、Exp36保持地点组内家庭关系，重排地点坐标。各次置换重新拟合表示并重新计算相应评估指标。</p>
    <p>跨seed的逐簇编号通过最大成员交集的一对一匹配对齐。图中的PCA为各seed分别拟合的二维投影；颜色在同一实验分支内对齐，黑圈标记Rain。各图的横纵坐标为各自PCA坐标。</p>
    <p>本报告中的地理留出指表示学习或回归阶段按地点划分；CEM策略向量沿用既有训练结果。行为回放使用既有家庭面板中的时间窗口。</p>'''))

    source_rows=[]
    for rel,path in SOURCES.items():
        encoded=base64.b64encode(path.read_bytes()).decode()
        filename=rel.replace('/','__')
        source_rows.append(f'<tr><td><code>{esc(rel)}</code></td><td><a href="data:text/csv;base64,{encoded}" download="{esc(filename)}">下载CSV</a></td></tr>')
    blocks.append(section("sources","11","结果文件与实验脚本",'''
    <p>实验结果根目录：<code>04.results/shape/</code>。实验脚本目录：<code>02.code/multiple_country/gps/</code>。</p>
    <div class="table-wrap"><table><thead><tr><th>实验</th><th>脚本</th></tr></thead><tbody>'''+''.join(f'<tr><td>{esc(a)}</td><td><code>{esc(b)}</code></td></tr>' for a,b in [("Exp32","gps_triplet_pilot.py"),("Exp33","gps_cluster_comparison.py"),("Exp34","gps_network_discovery.py；summarize_gps_network.py"),("Exp35","gps_place_networks.py"),("Exp36","gps_distance_regression.py"),("谱聚类对照","gps_spectral_comparison.py"),("Exp37","gps_behavior_increment.py"),("Exp38","audit_raw_small_cluster.py")])+'''</tbody></table></div>
    <details><summary>结果表索引与内嵌CSV下载</summary><div class="table-wrap"><table><thead><tr><th>相对结果路径</th><th>数据表</th></tr></thead><tbody>'''+''.join(source_rows)+'''</tbody></table></div></details>
    <p class="note">坐标来源：00.data/raw/家庭经纬度信息.xlsx；00.data/raw/副本家庭经纬度信息.xlsx。数值表从保存的实验结果读取，图片嵌入本文件。</p>'''))

    css='''
    :root{--ink:#172538;--muted:#596779;--blue:#173f70;--line:#dce2e9;--paper:#fff;--bg:#f4f6f9;--soft:#edf3fa}
    *{box-sizing:border-box}html{scroll-behavior:smooth;scroll-padding-top:85px}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.85 -apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",sans-serif}
    header{background:#122b49;color:white;padding:64px max(24px,calc((100vw - 1100px)/2)) 48px}header .eyebrow{font-size:12px;letter-spacing:.15em;color:#b6c9e2}h1{font-family:"Songti SC","Noto Serif SC",serif;font-size:38px;line-height:1.35;margin:14px 0 16px;letter-spacing:.02em}header p{margin:5px 0;color:#d4dfed;max-width:900px}header .meta{margin-top:24px;padding-top:18px;border-top:1px solid #49607b;display:flex;gap:12px 30px;flex-wrap:wrap;font-size:13px}
    nav{position:sticky;top:0;background:#fffffff2;border-bottom:1px solid var(--line);z-index:5;backdrop-filter:blur(10px)}nav .inner{max-width:1150px;margin:auto;display:flex;gap:5px;padding:10px 20px;overflow-x:auto}nav a{white-space:nowrap;text-decoration:none;color:var(--blue);font-size:13px;padding:4px 10px;border-radius:4px}nav a:hover{background:var(--soft)}main{max-width:1160px;padding:28px 30px 64px;margin:auto}section{background:var(--paper);border:1px solid var(--line);padding:34px 38px;margin-bottom:24px;border-radius:7px}.section-label{font:600 12px/1.3 ui-monospace,monospace;letter-spacing:.12em;color:#7290b2}h2{font-size:25px;line-height:1.45;margin:8px 0 22px;color:var(--blue)}h3{font-size:18px;margin:28px 0 10px}p{margin:14px 0}strong{font-weight:650}a{color:var(--blue)}.note{font-size:13px;color:var(--muted)}.facts{display:grid;grid-template-columns:repeat(4,1fr);margin:24px 0;background:var(--soft);border-radius:5px}.facts div{padding:18px 24px;border-right:1px solid #d4dfec}.facts div:last-child{border-right:0}.facts strong{display:block;font-size:30px;line-height:1.25;font-variant-numeric:tabular-nums;color:var(--blue)}.facts span{display:block;font-size:13px;color:var(--muted);margin-top:5px}.pipeline{display:flex;align-items:center;justify-content:center;gap:12px;flex-wrap:wrap;padding:20px;background:var(--soft);margin-bottom:24px;border-radius:5px}.pipeline span{padding:6px 9px;font-size:14px}.pipeline b{color:#6681a2}.table-wrap{overflow-x:auto;margin:20px 0}table{border-collapse:collapse;width:100%;font-size:13px;line-height:1.6;font-variant-numeric:tabular-nums}th,td{padding:11px 13px;border-bottom:1px solid var(--line);text-align:right;vertical-align:top}th{background:#ecf1f7;color:#243f61;font-size:12px;font-weight:650;white-space:normal}th:first-child,td:first-child{text-align:left}td{min-width:65px}tbody tr:nth-child(even){background:#fafbfd}caption{text-align:left;font-size:14px;font-weight:600;padding-bottom:10px}code{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;font-size:.85em;overflow-wrap:anywhere}.equation{font-family:Georgia,"Songti SC",serif;padding:22px;background:#f5f7fa;text-align:center;font-size:20px;border-left:3px solid #89a2c0}dl{font-size:14px;display:grid;grid-template-columns:165px 1fr;gap:12px 20px;margin:24px 0}dt{font-weight:600;color:var(--blue)}dd{margin:0}figure{margin:28px 0;border:1px solid #e4e8ee;border-radius:4px;padding:8px;background:#fff}figure img{display:block;width:100%;height:auto}figcaption{font-size:12px;line-height:1.7;color:var(--muted);padding:12px 10px 6px}details{margin:22px 0;border:1px solid var(--line);border-radius:4px;padding:14px 18px}summary{cursor:pointer;font-size:14px;font-weight:600;color:var(--blue)}footer{max-width:1100px;margin:0 auto;padding:0 30px 44px;color:var(--muted);font-size:12px;border-top:1px solid var(--line)}.overview-result{border-left:3px solid #6d8dac;padding:4px 18px;margin:18px 0;background:#f7f9fc}.overview-result p{font-size:15px}
    @media(max-width:700px){header{padding:38px 22px 30px}h1{font-size:28px}main{padding:16px 12px 40px}section{padding:25px 18px}h2{font-size:22px}.facts{grid-template-columns:repeat(2,1fr)}.facts div{border-bottom:1px solid #d4dfec}.facts strong{font-size:26px}dl{display:block}dt{margin-top:18px}dd{margin-top:5px}th,td{padding:9px 10px}.pipeline{gap:4px}}
    @media print{body{background:white;font-size:11pt}header{background:white!important;color:#172538;padding:0 0 20px}header p,header .eyebrow{color:#596779}header .meta{border-color:#dce2e9}nav{display:none}main{max-width:none;padding:0}section{border:none;padding:18px 0;margin:0;border-bottom:1px solid #ddd}h1{font-size:26pt}h2{font-size:18pt;break-after:avoid}h3{break-after:avoid}figure,.facts,tr{break-inside:avoid}thead{display:table-header-group}th{-webkit-print-color-adjust:exact;print-color-adjust:exact}details{border:none;padding:0}footer{padding:15px 0}#sources details{display:none}@page{size:A4;margin:16mm}}
    '''
    navigation=[('data','数据'),('methods','方法'),('exp32-33','Exp32–33'),('exp34','Exp34'),('exp35','Exp35'),('exp36','Exp36'),('spectral','谱聚类'),('exp37','Exp37'),('exp38','Exp38'),('appendix','补充结果'),('sources','文件索引')]
    content='''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>GPS辅助策略表示与聚类实验报告</title><style>'''+css+'''</style></head><body>
    <header><div class="eyebrow">HOUSEHOLD POLICY REPRESENTATIONS</div><h1>GPS 辅助策略表示与聚类<br>实验报告</h1><p>地理监督映射、跨种子成员复现与策略行为检验</p><div class="meta"><span>汇总日期：2026-11-07 · 更新版 v2</span><span>实验：Exp32–Exp38及谱聚类对照</span><span>样本：71户 · CEM seed 0／1／42</span></div></header>
    <nav><div class="inner">'''+''.join(f'<a href="#{a}">{b}</a>' for a,b in navigation)+'''</div></nav><main>
    <section><div class="section-label">RESULTS OVERVIEW</div><h2>结果概览</h2>
    <div class="overview-result"><p><strong>Exp35 同地点 SupCon：</strong>三个seed均输出4簇，四个匹配簇的三次共同成员为0、18、0、2户；18户共同成员全部属于Rain。</p></div>
    <div class="overview-result"><p><strong>Exp36 连续距离回归：</strong>HDBSCAN在三个seed均输出2簇，少数组为5、4、5户，三次共同成员1户。地理距离留出MAE为117.92 km，等权基准121.55 km，常数基准113.34 km。</p></div>
    <div class="overview-result"><p><strong>谱聚类对照：</strong>54次拟合中，Exp36在K=2、邻居参数5／10／15时的少数组三次共同成员分别为3／2／0户。</p></div>
    <div class="overview-result"><p><strong>Exp37 行为预测：</strong>主参数α=10时，三个行为指标加入经纬度后的总体RMSE均高于家庭特征模型。</p></div>
    <div class="overview-result"><p><strong>Exp38 原始θ小簇检查：</strong>小簇三次共同成员为10408、10495。全部213个保存策略的参数均未触及±1.5边界；各策略自身家庭的验证收益回放与日志一致。</p></div></section>
    '''+''.join(blocks)+'''</main><footer><p>GPS辅助策略表示与聚类实验报告 · 2026-11-07</p></footer>
    <script>window.addEventListener('beforeprint',()=>document.querySelectorAll('details').forEach(e=>{e.dataset.previousOpen=e.open?'1':'0';e.open=true}));window.addEventListener('afterprint',()=>document.querySelectorAll('details').forEach(e=>e.open=e.dataset.previousOpen==='1'));</script></body></html>'''
    OUT.write_text(content, encoding='utf-8')
    manifest={"report":str(OUT),"date":"2026-11-07","source_files":{rel:hashlib.sha256(p.read_bytes()).hexdigest() for rel,p in SOURCES.items()},"source_workbook_sha256":hashlib.sha256((ROOT/'00.data/raw/家庭经纬度信息.xlsx').read_bytes()).hexdigest()}
    (OUT.with_suffix('.sources.json')).write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(f"Saved {OUT} ({OUT.stat().st_size:,} bytes), {len(SOURCES)} source tables")


if __name__ == '__main__':
    main()
