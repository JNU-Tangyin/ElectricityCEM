"""Audit the raw-theta minority households against all other panel households.

No retraining, exclusions, or automatic causal diagnoses. Reconstruct the saved
Exp34 raw labels, audit input data and training histories, and replay checkpoint
profits under the existing train/validation-specific normalization convention.
"""
from pathlib import Path
from itertools import combinations
import argparse
import hashlib
import json
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import HDBSCAN
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / '02.code/multiple_country'))
from energy_storage_env import EnergyStorageEnv

SEEDS = ['cem_seed0', 'cem_seed1', 'cem_seed42']
SEED_ROOT = ROOT / '04.results/cem_npy/seed_wise'
DATA = ROOT / '00.data/preprocessed/german_households_weather'
KEEP = [i for i in range(36) if i not in (3, 35)]
EXPECTED = {
    'cem_seed0': {'10338', '10408', '10495', '10676', '11393'},
    'cem_seed1': {'10286', '10408', '10495', '12073'},
    'cem_seed42': {'10338', '10408', '10495', '10676', '11130', '11424', '11450'},
}


def markdown(df):
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return 'NA' if pd.isna(v) else f'{v:.5g}'
        return str(v)
    return '\n'.join(['| ' + ' | '.join(map(str, df.columns)) + ' |',
                      '| ' + ' | '.join(['---'] * len(df.columns)) + ' |'] +
                     ['| ' + ' | '.join(fmt(v) for v in row) + ' |'
                      for row in df.itertuples(index=False, name=None)])


def rollout(environment, theta, starts):
    profits = []
    for start in starts:
        obs, _ = environment.reset(options={'start_idx': int(start)})
        total, steps, done = 0., 0, False
        while not done:
            a = np.tanh(obs @ theta[:-1] + theta[-1])
            obs, _, done, _, info = environment.step(np.array([a]))
            total += info['profit']
            steps += 1
        profits.append(total / steps)
    return float(np.mean(profits))


def finding_text(data, params):
    core_data = data[data.core].copy()
    core_data['best_iter_seed0_1_42'] = core_data.household.map(
        lambda h: ' / '.join(str(int(params[(params.household == h) & (params.seed == s)].best_iter.iloc[0])) for s in SEEDS))
    cols = ['household','city','zero_load_fraction','pv_peak_wh','complete_day_load_mean_kwh','best_iter_seed0_1_42']
    ref = data[~data.candidate]
    cp = params[params.core]
    return f'''

## 核心户检查结果

{markdown(core_data[cols])}

- 候选10户零负荷比例中位数为 {data[data.candidate].zero_load_fraction.median():.2%}，参考61户中位数为 {ref.zero_load_fraction.median():.2%}；候选10户的零负荷比例均低于参考组中位数。
- 核心2户PV峰值为800、900 Wh，处于参考61户的 {ref.pv_peak_wh.min():.0f}–{ref.pv_peak_wh.max():.0f} Wh 范围内。
- 核心2户六个checkpoint的 best_iter 范围为 {int(cp.best_iter.min())}–{int(cp.best_iter.max())}（0起算）。
- 核心2户的34维标准化质心距离，均处于各自seed参考61户的 {cp.standardized_centroid_distance_pct_reference61.min():.2f}–{cp.standardized_centroid_distance_pct_reference61.max():.2f} 百分位。
- 以上检查没有出现“保存参数贴±1.5边界”“核心户零负荷比例很高”“核心户PV峰值超出参考组范围”这几种现象。记录完整与回放吻合不等于对传感器准确性的认证，也不能单独确定外围位置的成因。

![参数与输入数据分布](audit_distributions.png)

![核心户学习曲线](core_training_curves.png)
'''


def main(out):
    if out.exists():
        raise FileExistsError(f'Use a new --output directory: {out}')
    inputs = set()
    def source(p):
        inputs.add(p)
        return p
    saved = pd.read_csv(source(ROOT / '04.results/shape/exp34_gps_network_discovery/labels.csv'), dtype={'source': str})
    ids = sorted(p.name.removesuffix('_expert.npy') for p in (SEED_ROOT / SEEDS[0]).glob('*_expert.npy'))
    assert len(ids) == 71
    theta, standardized, labels, minority = {}, {}, {}, {}
    for seed in SEEDS:
        files = sorted((SEED_ROOT / seed).glob('*_expert.npy'))
        assert [p.name.removesuffix('_expert.npy') for p in files] == ids
        theta[seed] = np.stack([np.load(source(p)) for p in files])
        assert theta[seed].shape == (71, 36) and np.isfinite(theta[seed]).all()
        standardized[seed] = StandardScaler().fit_transform(theta[seed][:, KEEP])
        lab = HDBSCAN(min_cluster_size=4, min_samples=2).fit_predict(standardized[seed])
        ref = saved[(saved.seed == seed) & (saved.model == 'raw')].set_index('source').loc[ids, 'cluster'].to_numpy()
        assert np.array_equal(lab, ref), seed
        smallest = min(set(lab) - {-1}, key=lambda c: (lab == c).sum())
        minority[seed] = set(np.array(ids)[lab == smallest])
        assert minority[seed] == EXPECTED[seed], seed
        labels[seed] = lab
    candidates = set.union(*minority.values())
    core = set.intersection(*minority.values())
    assert len(candidates) == 10 and core == {'10408', '10495'}
    group = lambda h: 'core_2' if h in core else ('candidate_other_8' if h in candidates else 'reference_61')
    coordinates = pd.read_csv(source(ROOT / '00.data/csv/家庭经纬度信息.csv'), dtype={'id': str}).set_index('id')
    assert coordinates.index.is_unique
    meta_path = next(p for name in ['german_metadata.json', 'global_metadata.json', 'german_metadata_weather.json']
                     if (p := ROOT / '00.data/preprocessed' / name).exists())
    source(meta_path)
    frames, data_rows, parameter_rows, curves = {}, [], [], {}
    for seed in SEEDS:
        source(SEED_ROOT / seed / 'training_run_info.txt')
        summary = SEED_ROOT / seed / 'training_summary.csv'
        if summary.exists():
            source(summary)
    for j, hid in enumerate(ids):
        df = pd.read_csv(source(DATA / f'{hid}.csv'), index_col=0, parse_dates=True)
        frames[hid] = df
        assert len(df) == 287 and df.index.is_monotonic_increasing
        loc = coordinates.loc[hid]
        assert np.allclose(df[['lat', 'lon']].to_numpy(), [float(loc.latitude), float(loc.longitude)], rtol=0, atol=1e-12)
        daily = df.user_load_wh.resample('D').agg(['sum', 'count'])
        full_days = daily[daily['count'] == 24]
        data_rows.append(dict(household=hid, group=group(hid), core=hid in core,
            candidate=hid in candidates, minority_seed_count=sum(hid in m for m in minority.values()),
            city=loc.city, is_rain=loc.city == 'Rain', lat=loc.latitude, lon=loc.longitude,
            n_hours=len(df), start=str(df.index[0]), end=str(df.index[-1]),
            missing_load=int(df.user_load_wh.isna().sum()), missing_pv=int(df.pv_gen_wh.isna().sum()),
            negative_load=int((df.user_load_wh < 0).sum()), negative_pv=int((df.pv_gen_wh < 0).sum()),
            duplicate_timestamps=int(df.index.duplicated().sum()),
            nonhourly_gaps=int((df.index.to_series().diff().iloc[1:] != pd.Timedelta(hours=1)).sum()),
            zero_load_fraction=float((df.user_load_wh == 0).mean()),
            zero_pv_fraction=float((df.pv_gen_wh == 0).mean()),
            pv_peak_wh=float(df.pv_gen_wh.max()), load_peak_wh=float(df.user_load_wh.max()),
            load_mean_wh=float(df.user_load_wh.mean()),
            daily_load_equivalent_kwh=float(df.user_load_wh.mean() * 24 / 1000),
            complete_day_load_mean_kwh=float(full_days['sum'].mean() / 1000),
            n_complete_days=len(full_days)))
        train, val = df.iloc[:-96].copy(), df.iloc[-96:].copy()
        train_env = EnergyStorageEnv(train, episode_length_hours=72, sell_price_ratio=1., meta_path=str(meta_path))
        val_env = EnergyStorageEnv(val, episode_length_hours=95, sell_price_ratio=1., meta_path=str(meta_path))
        last = len(train) - 72 - 1
        starts = [0, last // 2, last]
        for seed in SEEDS:
            w = theta[seed][j]
            hist = pd.read_csv(source(SEED_ROOT / seed / f'{hid}_history.csv'))
            assert len(hist) == 300 and np.isfinite(hist[['mean_reward', 'eval_profit']]).all().all()
            best = int(np.argmax(hist.eval_profit.to_numpy()))
            curves[(seed, hid)] = hist
            replay_val = rollout(val_env, w, [0])
            replay_train = rollout(train_env, w, starts)
            z = standardized[seed]
            row = dict(seed=seed, household=hid, group=group(hid), core=hid in core,
                candidate=hid in candidates, in_small_cluster=hid in minority[seed], cluster=int(labels[seed][j]),
                frac_at_bound=float((np.abs(w) >= 1.5 - 1e-6).mean()),
                frac_abs_ge_1_4=float((np.abs(w) >= 1.4).mean()), max_abs_weight=float(np.abs(w).max()),
                w_norm=float(np.linalg.norm(w[:-1])), geometry34_norm=float(np.linalg.norm(w[KEEP])),
                bias=float(w[-1]), standardized_centroid_distance=float(np.linalg.norm(z[j] - z.mean(axis=0))),
                history_rows=len(hist), best_iter=int(hist.iloc[best].iteration),
                best_iter_fraction=float(best / (len(hist)-1)),
                logged_best_val_profit_eur_h=float(hist.iloc[best].eval_profit),
                logged_final_val_profit_eur_h=float(hist.iloc[-1].eval_profit),
                logged_population_train_mean_at_best_eur_h=float(hist.iloc[best].mean_reward),
                replay_checkpoint_train_fixed3_eur_h=replay_train, replay_checkpoint_val_eur_h=replay_val,
                val_replay_abs_error=abs(replay_val - float(hist.iloc[best].eval_profit)),
                has_parameter_trajectory=False, checkpoint_trajectory_max_error=np.nan,
                trajectory_step_first10=np.nan, trajectory_step_last50=np.nan)
            traj_path = SEED_ROOT / seed / f'{hid}_traj_mu.npy'
            if traj_path.exists():
                traj = np.load(source(traj_path))
                assert traj.shape == (len(hist), 36)
                steps = np.linalg.norm(np.diff(traj[:, KEEP], axis=0), axis=1)
                row.update(has_parameter_trajectory=True,
                    checkpoint_trajectory_max_error=float(np.max(np.abs(w-traj[best]))),
                    trajectory_step_first10=float(steps[:10].mean()),
                    trajectory_step_last50=float(steps[-50:].mean()))
            parameter_rows.append(row)
        train_env.close()
        val_env.close()
        if (j+1) % 15 == 0:
            print(f'Audited {j+1}/71 households', flush=True)
    data = pd.DataFrame(data_rows)
    params = pd.DataFrame(parameter_rows)
    assert params.val_replay_abs_error.max() < 1e-9, 'Checkpoint replay differs from saved history'
    assert params.checkpoint_trajectory_max_error.dropna().max() < 1e-12
    metrics = ['frac_at_bound', 'max_abs_weight', 'w_norm', 'bias', 'standardized_centroid_distance',
               'best_iter', 'replay_checkpoint_train_fixed3_eur_h', 'replay_checkpoint_val_eur_h']
    for metric in metrics:
        params[metric + '_pct_reference61'] = params.groupby('seed')[metric].transform(
            lambda x: x.map(lambda value: 100 * ((x[params.loc[x.index, 'group'] == 'reference_61'] < value).mean() +
                                                .5 * (x[params.loc[x.index, 'group'] == 'reference_61'] == value).mean())))
    data_metrics = ['zero_load_fraction', 'pv_peak_wh', 'complete_day_load_mean_kwh']
    for metric in data_metrics:
        reference = data.loc[~data.candidate, metric].to_numpy()
        data[metric + '_pct_reference61'] = data[metric].map(lambda v: 100*((reference < v).mean()+.5*(reference == v).mean()))
    summary = []
    for scope, table, fields in [('data', data, data_metrics), *[(s, params[params.seed == s], metrics) for s in SEEDS]]:
        for name, d in table.groupby('group'):
            for field in fields:
                v = d[field]
                summary.append(dict(scope=scope, group=name, metric=field, n=len(v),
                    mean=v.mean(), median=v.median(), q25=v.quantile(.25), q75=v.quantile(.75), min=v.min(), max=v.max()))
    overlap = pd.DataFrame([dict(seed_a=a, seed_b=b, intersection=len(minority[a]&minority[b]),
        union=len(minority[a]|minority[b]), jaccard=len(minority[a]&minority[b])/len(minority[a]|minority[b]))
        for a, b in combinations(SEEDS, 2)])
    # Same-household retrieval is supplemental; a low identity score is not a test of absence of types.
    pooled = np.stack([theta[s][:, KEEP] for s in SEEDS], axis=1)
    pooled = StandardScaler().fit_transform(pooled.reshape(-1, 34)).reshape(71, 3, 34)
    retrieval = []
    for a, b in [(a,b) for a in range(3) for b in range(3) if a != b]:
        distances = cdist(pooled[:, a], pooled[:, b])
        for i, h in enumerate(ids):
            rank = int(1 + np.sum(distances[i] < distances[i,i]))
            retrieval.append(dict(seed_query=SEEDS[a], seed_gallery=SEEDS[b], household=h,
                                  group=group(h), rank=rank, top1=rank <= 1, top5=rank <= 5,
                                  same_household_distance=distances[i,i],
                                  other_household_mean_distance=np.delete(distances[i],i).mean()))
    retrieval = pd.DataFrame(retrieval)
    out.mkdir(parents=True)
    joined = params.merge(data.drop(columns=['group', 'core', 'candidate']), on='household', validate='many_to_one')
    for name, df in [('all_households_data', data), ('all_households_parameters_training', params),
                     ('candidate_10_audit', joined[joined.candidate]), ('core_2_audit', joined[joined.core]),
                     ('group_summary', pd.DataFrame(summary)), ('small_cluster_overlap', overlap),
                     ('cross_seed_retrieval', retrieval)]:
        df.to_csv(out / f'{name}.csv', index=False)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), layout='constrained')
    panels = [(data, data_metrics[0]), (data, data_metrics[1]), (data, data_metrics[2]),
              (params, 'max_abs_weight'), (params, 'best_iter'), (params, 'standardized_centroid_distance')]
    for ax, (df, metric) in zip(axes.flat, panels):
        names = ['reference_61','candidate_other_8','core_2']
        ax.boxplot([df[df.group == g][metric] for g in names], tick_labels=['Other 61','Candidate 8','Core 2'], showfliers=False)
        for k, name in enumerate(names):
            vals = df[df.group == name][metric].to_numpy()
            ax.scatter(k+1+np.linspace(-.12,.12,len(vals)), vals, s=10, alpha=.55)
        ax.set_title(metric, fontsize=10)
        if metric == 'max_abs_weight':
            ax.axhline(1.5, ls='--', color='red', lw=1)
    fig.suptitle('Raw-theta minority audit | bottom row pools 3 seeds descriptively')
    fig.savefig(out / 'audit_distributions.png', dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), layout='constrained')
    for col, seed in enumerate(SEEDS):
        for row, metric in enumerate(['eval_profit','mean_reward']):
            ax = axes[row, col]
            refs = np.stack([curves[(seed,h)][metric] for h in ids if h not in candidates])
            ax.fill_between(np.arange(300), *np.quantile(refs,[.25,.75],axis=0), color='gray', alpha=.2, label='Other 61 IQR')
            ax.plot(np.median(refs,axis=0), color='gray', lw=1, label='Other 61 median')
            for h, color in [('10408','#0072B2'), ('10495','#D55E00')]:
                ax.plot(curves[(seed,h)][metric], color=color, lw=1, label=h)
            ax.set(title=f'{seed}: {metric}', xlabel='Iteration (0-based)', ylabel='Benefit-Cost EUR/h')
    axes[0,0].legend(fontsize=8)
    fig.savefig(out / 'core_training_curves.png', dpi=170)
    plt.close(fig)
    brief = data[data.candidate][['household','minority_seed_count','city','zero_load_fraction','pv_peak_wh','complete_day_load_mean_kwh']].copy()
    for seed in SEEDS:
        brief[seed+'_best_iter'] = brief.household.map(params[params.seed==seed].set_index('household').best_iter)
    text = f'''# Raw θ 小簇候选家庭检查

## 范围与复现

71 户 × 3 个 CEM seed；原始策略 36 维，HDBSCAN 使用逐 seed 标准化的 34 维（去掉位置索引3与偏置35），参数 min_cluster_size=4、min_samples=2。标签与 Exp34 保存结果逐项相同。候选并集10户、三次交集2户：10408、10495。参考组为候选并集之外的61户，包含个别 seed 的噪声点。

{markdown(overlap)}

## 10户数据与最佳迭代

{markdown(brief)}

日负荷单位 kWh/day，仅使用完整24小时日求均值；PV峰值为小时记录 Wh。详细表另列全窗小时均值×24及完整天数。数据来源为实际训练使用的预处理 CSV，不等同于对采集器原始记录的质量认证。

## 参数与训练核对

- 全部213个checkpoint中，贴 ±1.5 边界的维度总数：{int(sum((np.abs(w)>=1.5-1e-6).sum() for w in theta.values()))}；最大绝对参数：{params.max_abs_weight.max():.9f}。这检查的是保存的分布中心，不代表训练候选种群从未被 clip。
- 验证回放与历史最高验证收益最大绝对误差：{params.val_replay_abs_error.max():.3g} EUR/h。
- 参数轨迹可用：{int(params.has_parameter_trajectory.sum())}/213；可用轨迹的最佳代与checkpoint最大误差：{params.checkpoint_trajectory_max_error.max():.3g}。
- best_iter 为日志 eval_profit 首次最大值所在迭代（0起算）。较早取得最佳验证值不是未收敛的充分证据。
- train 固定回放使用前191小时、三个72小时窗口（起点0、59、118）；val使用最后96小时中的95步，各自使用对应分段的缩放，与当前训练脚本一致。训练回放衡量保存θ，不等同于原训练随机窗口下的候选种群均值。
- logged_population_train_mean_at_best_eur_h 是最佳代的种群训练均值，不能当成该checkpoint训练收益。所有收益均为现有 Benefit-Cost 口径，不命名为电费节省。

## 数据检查

全体71户负荷/PV缺失值合计 {int(data[['missing_load','missing_pv']].to_numpy().sum())}；负值合计 {int(data[['negative_load','negative_pv']].to_numpy().sum())}；重复时间戳 {int(data.duplicate_timestamps.sum())}；非小时间隔 {int(data.nonhourly_gaps.sum())}。零负荷比例和PV峰值属于描述性检查，不能单独确定异常原因。

## 跨seed补充检索

用全部213条记录拟合共同标准化尺度，去除位置与偏置；每个查询在另一seed的71户中检索。六个有方向seed配对平均 top-1={retrieval.top1.mean():.3%}，top-5={retrieval.top5.mean():.3%}。等概率参考分别为1/71、5/71；未以此作显著性或停止聚类的判据。六个配对重复使用同一71户，不视为426个独立家庭。

既有户内/户间距离置换检验见 exp1b_variance_decomposition/within_between_distance.csv；本次不重复拟合该检验，不把仅有2户稳定进入最小簇解释为只有2户具备任何家庭信号。

## 文件

- candidate_10_audit.csv：10户×3 seed，参数、训练、坐标及输入数据指标。
- core_2_audit.csv：2户×3 seed。
- all_households_data.csv / all_households_parameters_training.csv：71户及213个策略的完整对照。
- *_pct_reference61：相对同seed参考61户的中秩百分位；不作p值或异常阈值。
- group_summary.csv：核心2户、其余候选8户、参考61户的分布；逐seed列出，避免将重复训练当独立家庭。
- audit_distributions.png / core_training_curves.png：分布图与核心户学习曲线。
- cross_seed_retrieval.csv：逐户检索结果；small_cluster_overlap.csv：小簇交集。
- manifest.json：输入与脚本哈希及执行协议。

本检查不自动把家庭归因为训练故障、测量故障或真实类型；候选名单由θ选出，其参数范数或质心距离差异属于选择后描述。
'''
    (out / 'REPORT.md').write_text(text + finding_text(data, params))
    for p in [Path(__file__), ROOT/'02.code/multiple_country/train_cem.py', ROOT/'02.code/multiple_country/energy_storage_env.py']:
        source(p)
    manifest = dict(protocol='raw_small_cluster_audit_v1', seeds=SEEDS, n_households=71,
        candidates=sorted(candidates), core=sorted(core), max_w=1.5, boundary_tolerance=1e-6,
        geometry_indices=KEEP, training_replay_starts=[0,59,118], validation_steps=95,
        files=[dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(inputs)])
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False))
    print(brief.to_string(index=False))
    print('Maximum validation replay error:', params.val_replay_abs_error.max())
    print('Saved', out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'04.results/shape/exp38_raw_small_cluster_audit')
    with threadpool_limits(limits=1):
        main(parser.parse_args().output)
