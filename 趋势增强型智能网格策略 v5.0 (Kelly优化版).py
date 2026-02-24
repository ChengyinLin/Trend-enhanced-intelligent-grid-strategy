import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from collections import deque
import warnings
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


@dataclass(slots=True)
class Trade:
    """交易记录"""
    date: pd.Timestamp
    type: str
    price: float
    shares: float
    value: float
    pnl: Optional[float] = None
    grid_level: Optional[int] = None


@dataclass
class StrategyConfig:
    """策略配置参数 - v5.0 Kelly优化版"""
    initial_capital: float = 100000.0
    base_grid_pct: float = 0.03          # 降低基础间距
    grid_levels: int = 7                 # 增加层级
    max_position: float = 0.95           # 提高上限
    stop_loss: float = 0.15              # 放宽止损
    trend_period: int = 20
    vol_period: int = 14
    cooldown_days: int = 8               # 缩短冷却期
    min_grid_spacing: float = 0.015
    max_grid_spacing: float = 0.10       # 扩大上限
    position_update_threshold: float = 0.06  # 更敏感
    min_cash_reserve: float = 0.05       # 降低现金储备
    weekly_trend_period: int = 20
    atr_stop_multiplier: float = 2.5
    rebalance_interval: int = 20
    
    # Kelly专用参数
    kelly_lookback: int = 40             # Kelly回望窗口
    kelly_fraction: float = 0.25         # 1/4 Kelly保守系数
    kelly_max_position: float = 0.25     # 单标的上限
    
    # 波动率聚类参数
    vol_cluster_window: int = 20         # 波动率记忆窗口
    vol_cluster_threshold: float = 0.3   # 聚类检测阈值
    
    # 多时间框架参数
    mtf_periods: Dict[str, int] = field(default_factory=lambda: {
        'short': 5, 'medium': 20, 'long': 60
    })


class KellyPositionSizer:
    """Kelly准则动态仓位管理器 - 贝叶斯优化版"""
    
    def __init__(self, config: StrategyConfig):
        self.config = config
        self.trade_history: deque = deque(maxlen=config.kelly_lookback)
        self.current_kelly: float = 0.10   # 默认初始值
        self.win_streak: int = 0
        self.loss_streak: int = 0
        self.last_update_idx: int = 0
        
    def update_trade(self, pnl: float, entry_price: float, exit_price: float, 
                     holding_days: int, trend_score: float, idx: int):
        """记录交易并更新Kelly估计"""
        if idx <= self.last_update_idx:
            return
        self.last_update_idx = idx
            
        ret = (exit_price - entry_price) / entry_price if entry_price > 0 else 0
        
        self.trade_history.append({
            'pnl': pnl,
            'return': ret,
            'days': holding_days,
            'trend': trend_score,
            'timestamp': idx
        })
        
        # 更新连赢/连输计数
        if pnl > 0:
            self.win_streak += 1
            self.loss_streak = 0
        else:
            self.loss_streak += 1
            self.win_streak = 0
        
        self._recalculate_kelly()
    
    def _recalculate_kelly(self):
        """使用贝叶斯更新计算Kelly比例"""
        if len(self.trade_history) < 8:
            self.current_kelly = 0.08
            return
            
        returns = np.array([t['return'] for t in self.trade_history])
        
        # 计算胜率 W 和盈亏比 R
        wins = returns[returns > 0]
        losses = returns[returns < 0]
        
        W = len(wins) / len(returns) if len(returns) > 0 else 0.5
        
        if len(losses) > 0 and np.mean(losses) != 0:
            R = np.mean(wins) / abs(np.mean(losses)) if len(wins) > 0 else 1.0
        else:
            R = 2.0  # 默认盈亏比
            
        # Kelly公式: f* = (W*R - (1-W)) / R
        if R > 0:
            kelly_raw = (W * R - (1 - W)) / R
            # 应用分数Kelly并限制范围
            kelly_adj = kelly_raw * self.config.kelly_fraction
            
            # 根据连赢/连输调整（趋势跟踪）
            if self.win_streak >= 3:
                kelly_adj *= 1.15  # 连赢时增加15%
            elif self.loss_streak >= 2:
                kelly_adj *= 0.85  # 连输时减少15%
                
            self.current_kelly = np.clip(kelly_adj, 0.02, self.config.kelly_max_position)
        else:
            self.current_kelly = 0.05
    
    def get_position_size(self, total_value: float, price: float, 
                         volatility_regime: str, trend_strength: float,
                         alignment_score: float) -> Tuple[float, float]:
        """
        根据市场状态调整Kelly仓位
        返回: (shares, position_pct)
        """
        if total_value <= 0 or price <= 0:
            return 0.0, 0.0
            
        base_size = self.current_kelly
        
        # 波动率状态调整
        vol_multipliers = {
            'low': 1.3,           # 低波动激进
            'low_clustering': 1.1,
            'normal': 1.0,
            'high_clustering': 0.6,  # 高波动聚类保守
            'high': 0.5
        }
        vol_mult = vol_multipliers.get(volatility_regime, 1.0)
        
        # 多时间框架一致性调整
        alignment_mult = 1 + abs(alignment_score) * 0.4  # 一致性越高，仓位越大
        
        # 趋势强度调整
        trend_mult = 1 + max(0, trend_strength) * 0.3
        
        adjusted_pct = base_size * vol_mult * alignment_mult * trend_mult
        adjusted_pct = min(adjusted_pct, self.config.kelly_max_position)
        
        # 计算实际股数（保留现金缓冲）
        invest_amount = total_value * adjusted_pct * 0.98  # 2%缓冲
        shares = invest_amount / price
        
        return shares, adjusted_pct


class VolatilityClusteringGrid:
    """波动率聚类自适应网格 - GARCH-like检测"""
    
    def __init__(self, config: StrategyConfig):
        self.config = config
        self.volatility_memory: deque = deque(maxlen=config.vol_cluster_window)
        self.price_memory: deque = deque(maxlen=config.vol_cluster_window)
        self.regime: str = 'normal'
        self.clustering_factor: float = 1.0
        self.regime_duration: int = 0
        
    def detect_regime(self, current_atr: float, price: float, idx: int) -> str:
        """检测波动率聚类状态"""
        if price <= 0 or current_atr <= 0:
            return self.regime
            
        vol_pct = current_atr / price
        self.volatility_memory.append(vol_pct)
        self.price_memory.append(price)
        
        if len(self.volatility_memory) < 10:
            return 'normal'
            
        vol_series = pd.Series(self.volatility_memory)
        current_vol = vol_series.iloc[-1]
        mean_vol = vol_series.mean()
        
        # 计算波动率的波动率（聚类检测）
        vol_of_vol = vol_series.rolling(5).std().iloc[-1]
        vol_trend = (current_vol / mean_vol - 1) if mean_vol > 0 else 0
        
        # 价格动量（辅助判断）
        if len(self.price_memory) >= 5:
            price_series = pd.Series(self.price_memory)
            price_change = abs(price_series.iloc[-1] / price_series.iloc[-5] - 1)
        else:
            price_change = 0
            
        # GARCH-like 聚类检测逻辑
        new_regime = self.regime
        
        if vol_trend > self.config.vol_cluster_threshold and vol_of_vol > mean_vol * 0.4:
            # 波动率上升且波动率的波动率高 = 高波动聚类
            new_regime = 'high_clustering'
            self.clustering_factor = 1.6
        elif vol_trend < -0.25 and current_vol < mean_vol * 0.8:
            # 波动率持续下降 = 低波动聚类
            new_regime = 'low_clustering'
            self.clustering_factor = 0.75
        elif current_vol > mean_vol * 1.5:
            new_regime = 'high'
            self.clustering_factor = 1.3
        elif current_vol < mean_vol * 0.6:
            new_regime = 'low'
            self.clustering_factor = 0.85
        else:
            new_regime = 'normal'
            self.clustering_factor = 1.0
            
        # 状态持续时间计数（避免频繁切换）
        if new_regime == self.regime:
            self.regime_duration += 1
        else:
            # 状态切换需要至少3天确认
            if self.regime_duration < 3 and self.regime != 'normal':
                pass  # 保持原状态
            else:
                self.regime = new_regime
                self.regime_duration = 0
                
        return self.regime
    
    def get_dynamic_spacing(self, base_spacing: float, atr: float, price: float) -> float:
        """根据聚类状态调整网格间距"""
        adjusted = base_spacing * self.clustering_factor
        
        # 在高波动聚类期间使用更宽的网格
        if self.regime == 'high_clustering':
            adjusted *= (1 + atr / price * 3)
            
        return max(self.config.min_grid_spacing, 
                  min(self.config.max_grid_spacing, adjusted))
    
    def should_widen_grids(self) -> bool:
        """判断是否应该扩大网格（避免频繁交易）"""
        return self.regime in ['high_clustering', 'high'] and self.regime_duration >= 3


class MultiTimeframeMomentum:
    """多时间框架动量共振系统"""
    
    def __init__(self, df: pd.DataFrame, config: StrategyConfig):
        self.df = df
        self.config = config
        self.timeframes = config.mtf_periods
        self._precompute()
    
    def _precompute(self):
        """预计算多时间框架指标"""
        for name, period in self.timeframes.items():
            self.df[f'MA_{name}'] = self.df['Close'].rolling(period).mean()
            self.df[f'EMA_{name}'] = self.df['Close'].ewm(span=period).mean()
            
            # 动量
            self.df[f'Momentum_{name}'] = (
                self.df['Close'] / self.df[f'MA_{name}'] - 1
            )
            
            # 波动率
            self.df[f'Volatility_{name}'] = (
                self.df['Close'].pct_change().rolling(period).std()
            )
            
            # 趋势强度（ADX-like）
            plus_dm = self.df['High'].diff()
            minus_dm = self.df['Low'].diff(-1).abs()
            self.df[f'TrendStrength_{name}'] = (
                (plus_dm > minus_dm) & (plus_dm > 0)
            ).rolling(period).mean()
    
    def get_composite_signal(self, idx: int) -> Dict:
        """计算复合动量信号"""
        weights = {'short': 0.5, 'medium': 0.3, 'long': 0.2}
        
        composite = 0
        alignment = 0
        signals = {}
        volatilities = []
        
        for name in self.timeframes.keys():
            mom = self.df[f'Momentum_{name}'].iloc[idx]
            signals[name] = mom
            composite += mom * weights[name]
            
            # 方向一致性
            alignment += 1 if mom > 0 else -1
            volatilities.append(self.df[f'Volatility_{name}'].iloc[idx])
            
        # 标准化一致性得分 (-1 到 1)
        alignment_score = alignment / len(self.timeframes)
        
        # 趋势质量（基于波动率调整的信噪比）
        avg_vol = np.mean(volatilities) if volatilities else 0.01
        snr = abs(composite) / (avg_vol + 1e-6)
        
        # 趋势持续性（短期与长期相关性）
        persistence = 1.0
        if len(signals) >= 2:
            short_sig = list(signals.values())[0]
            long_sig = list(signals.values())[-1]
            if abs(long_sig) > 0.01:
                persistence = 1 if short_sig * long_sig > 0 else 0.5
                
        return {
            'composite': composite,
            'alignment': alignment_score,
            'strength': snr,
            'persistence': persistence,
            'signals': signals,
            'quality': snr * persistence  # 综合质量得分
        }
    
    def should_trade(self, idx: int, direction: str = 'long') -> Tuple[bool, float, float]:
        """
        多时间框架确认是否可以交易
        返回: (should_trade, strength, alignment)
        """
        data = self.get_composite_signal(idx)
        
        # 强一致性过滤：至少两个周期同向 (alignment >= 0.33)
        if abs(data['alignment']) < 0.33:
            return False, 0.0, data['alignment']
            
        # 趋势质量过滤
        if data['quality'] < 0.3:
            return False, 0.0, data['alignment']
            
        # 方向判断
        if direction == 'long':
            if data['composite'] > 0 and data['alignment'] > 0:
                return True, data['strength'], data['alignment']
        else:
            if data['composite'] < 0 and data['alignment'] < 0:
                return True, data['strength'], data['alignment']
                
        return False, 0.0, data['alignment']


class SmartRebalanceTrigger:
    """基于信息熵和市场微观结构的智能再平衡"""
    
    def __init__(self, threshold: float = 0.12):
        self.threshold = threshold
        self.price_history: deque = deque(maxlen=60)
        self.returns_history: deque = deque(maxlen=30)
        self.last_rebalance_idx = 0
        self.entropy_history: deque = deque(maxlen=10)
        
    def calculate_price_entropy(self) -> float:
        """计算价格分布的信息熵"""
        if len(self.returns_history) < 15:
            return 0.0
            
        returns = np.array(self.returns_history)
        
        # 自适应分箱
        n_bins = min(10, len(returns) // 3)
        hist, _ = np.histogram(returns, bins=n_bins, density=True)
        hist = hist[hist > 0]
        
        if len(hist) == 0:
            return 0.0
            
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        max_entropy = np.log2(n_bins)
        
        return entropy / max_entropy if max_entropy > 0 else 0  # 标准化到0-1
    
    def detect_regime_shift(self, current_idx: int) -> bool:
        """检测市场状态转变"""
        if len(self.entropy_history) < 5:
            return False
            
        recent_entropy = np.mean(list(self.entropy_history)[-3:])
        prev_entropy = np.mean(list(self.entropy_history)[:-3])
        
        # 熵值突变表示状态转变
        return abs(recent_entropy - prev_entropy) > 0.2
    
    def check_rebalance(self, current_idx: int, price: float, 
                       grid_manager: Optional['GridManager']) -> Tuple[bool, str]:
        """
        检查是否需要再平衡
        返回: (should_rebalance, reason)
        """
        self.price_history.append(price)
        
        if len(self.price_history) >= 2:
            ret = (price - self.price_history[-2]) / self.price_history[-2]
            self.returns_history.append(ret)
        
        # 最小间隔检查
        if current_idx - self.last_rebalance_idx < 5:
            return False, ""
            
        if len(self.price_history) < 20:
            return False, ""
            
        # 计算当前偏离度
        if grid_manager:
            center = grid_manager.center_price
            deviation = abs(price - center) / center if center > 0 else 0
            
            # 价格熵计算
            entropy = self.calculate_price_entropy()
            self.entropy_history.append(entropy)
            
            # 触发条件1：价格偏离过大
            if deviation > self.threshold:
                self.last_rebalance_idx = current_idx
                return True, "price_deviation"
                
            # 触发条件2：信息熵突变（状态转变）
            if self.detect_regime_shift(current_idx) and entropy > 0.6:
                self.last_rebalance_idx = current_idx
                return True, "regime_shift"
                
            # 触发条件3：网格失效检测（价格长期不在任何网格区间）
            grid_range = (grid_manager.grids[-1]['buy_price'], 
                         grid_manager.grids[0]['sell_price'])
            if price < grid_range[0] * 0.95 or price > grid_range[1] * 1.05:
                self.last_rebalance_idx = current_idx
                return True, "grid_invalid"
                
        return False, ""


class GridManager:
    """增强版网格管理器 - 支持动态层级调整"""
    
    def __init__(self, center_price: float, levels: int, spacing: float):
        self.center_price = center_price
        self.levels = levels
        self.spacing = spacing
        self.grids: List[Dict] = []
        self._init_grids()
    
    def _init_grids(self):
        """初始化网格"""
        self.grids = []
        for i in range(1, self.levels + 1):
            self.grids.append({
                'level': i,
                'buy_price': self.center_price * (1 - self.spacing * i),
                'sell_price': self.center_price * (1 + self.spacing * i),
                'active': False,
                'shares': 0.0,
                'entry_price': 0.0,
                'entry_date': None,
                'kelly_pct': 0.0  # 记录建仓时的Kelly比例
            })
    
    def update_center(self, new_center: float, spacing: float, 
                     preserve_active: bool = False) -> List[Dict]:
        """更新网格中心，支持动态层级"""
        closed_grids = []
        
        if preserve_active:
            old_grids = {g['level']: g for g in self.grids if g['active']}
            old_center = self.center_price
            
            price_ratio = new_center / old_center if old_center > 0 else 1.0
            
            self.center_price = new_center
            self.spacing = spacing
            
            # 动态调整层级：根据间距大小调整
            if spacing > 0.05:
                new_levels = max(4, self.levels - 1)  # 宽间距减少层级
            elif spacing < 0.025:
                new_levels = min(9, self.levels + 1)  # 窄间距增加层级
            else:
                new_levels = self.levels
                
            self.levels = new_levels
            self._init_grids()
            
            for level, old_grid in old_grids.items():
                if level > self.levels:
                    closed_grids.append(old_grid)
                    continue
                
                new_grid = self.grids[level - 1]
                adjusted_entry = old_grid['entry_price'] * price_ratio
                
                # 盈利安全检查
                min_profit_price = adjusted_entry * 1.002
                
                if new_grid['sell_price'] <= min_profit_price:
                    closed_grids.append(old_grid)
                    continue
                
                new_grid['active'] = True
                new_grid['shares'] = old_grid['shares']
                new_grid['entry_price'] = adjusted_entry
                new_grid['entry_date'] = old_grid.get('entry_date')
                new_grid['kelly_pct'] = old_grid.get('kelly_pct', 0)
        else:
            closed_grids = [g for g in self.grids if g['active']]
            self.center_price = new_center
            self.spacing = spacing
            self._init_grids()
        
        return closed_grids
    
    def get_active_grids(self) -> List[Dict]:
        return [g for g in self.grids if g['active']]
    
    def get_inactive_buy_grids(self) -> List[Dict]:
        inactive = [g for g in self.grids if not g['active']]
        return sorted(inactive, key=lambda x: x['buy_price'], reverse=True)
    
    def check_signals(self, high: float, low: float) -> Tuple[List[Dict], List[Dict]]:
        buy_signals = []
        sell_signals = []
        
        for grid in self.grids:
            if not grid['active'] and low <= grid['buy_price']:
                buy_signals.append(grid)
            elif grid['active'] and high >= grid['sell_price']:
                sell_signals.append(grid)
        
        return buy_signals, sell_signals
    
    def get_nearest_grid_distance(self, price: float) -> float:
        """获取价格到最近网格的距离比例"""
        if not self.grids:
            return 1.0
            
        distances = []
        for g in self.grids:
            if not g['active']:
                distances.append(abs(price - g['buy_price']) / price)
            else:
                distances.append(abs(price - g['sell_price']) / price)
                
        return min(distances) if distances else 1.0


class TrendEnhancedGridStrategyV5:
    """
    趋势增强型智能网格策略 v5.0
    核心优化：Kelly仓位管理 + 波动率聚类 + 多时间框架共振
    """
    
    def __init__(self, config: Optional[StrategyConfig] = None):
        self.config = config or StrategyConfig()
        self.reset()
    
    def reset(self):
        self.cash = self.config.initial_capital
        self.position = 0.0
        self.cost_basis = 0.0
        self.grid_manager: Optional[GridManager] = None
        self.trades: List[Trade] = []
        self.values: List[Dict] = []
        self.max_value = self.config.initial_capital
        self.cooldown_counter = 0
        self.last_grid_update_idx = 0
        self._last_add_idx = 0
        self._entry_max_price = 0.0
        
        # 重启状态
        self._restart_phase = 0
        self._restart_base_cash = 0.0
        self._last_restart_idx = 0
        
        # 新增组件
        self.kelly_sizer = KellyPositionSizer(self.config)
        self.vol_cluster = VolatilityClusteringGrid(self.config)
        self.rebalance_trigger = SmartRebalanceTrigger()
        self.mtf_momentum: Optional[MultiTimeframeMomentum] = None
        
        self._df: Optional[pd.DataFrame] = None
        self._current_idx: int = 0
    
    def prepare_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """预计算所有技术指标"""
        df = df.copy()
        
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values('datetime').reset_index(drop=True)
        
        # 处理成交量
        if df['Volume'].dtype == object:
            def parse_volume(v):
                if isinstance(v, str):
                    v = v.strip()
                    if 'B' in v.upper(): 
                        return float(v.replace('B', '').replace('b', '')) * 1e9
                    elif 'M' in v.upper(): 
                        return float(v.replace('M', '').replace('m', '')) * 1e6
                    elif 'K' in v.upper():
                        return float(v.replace('K', '').replace('k', '')) * 1e3
                return float(v)
            df['Volume'] = df['Volume'].apply(parse_volume)
        
        # 基础指标
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        high_low = df['High'] - df['Low']
        high_close = (df['High'] - df['Close'].shift()).abs()
        low_close = (df['Low'] - df['Close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(window=self.config.vol_period).mean()
        
        df['MA5'] = df['Close'].rolling(5).mean()
        df['MA10'] = df['Close'].rolling(10).mean()
        df['MA20'] = df['Close'].rolling(self.config.trend_period).mean()
        
        # 周线趋势
        df['Weekly_MA'] = df['Close'].rolling(self.config.weekly_trend_period).mean()
        df['Weekly_MA_Slow'] = df['Weekly_MA'].rolling(4).mean()
        df['Weekly_Trend'] = np.where(df['Weekly_MA'] > df['Weekly_MA_Slow'], 1, -1)
        
        # 日线趋势评分
        conditions = [
            (df['Close'] > df['MA5']) & (df['MA5'] > df['MA10']) & (df['MA10'] > df['MA20']),
            (df['Close'] < df['MA5']) & (df['MA5'] < df['MA10']) & (df['MA10'] < df['MA20'])
        ]
        
        bull_score = ((df['Close'] / df['MA20'] - 1) * 5).clip(upper=1.0)
        bear_score = ((df['Close'] / df['MA20'] - 1) * 5).clip(lower=-1.0)
        range_score = ((df['Close'] - df['MA20']) / df['MA20'] * 3).clip(-0.3, 0.3)
        
        df['Trend'] = np.select(conditions, [bull_score, bear_score], default=range_score)
        
        rsi_overbought = df['RSI'] > 75
        rsi_oversold = df['RSI'] < 25
        df.loc[rsi_overbought, 'Trend'] -= 0.3
        df.loc[rsi_oversold, 'Trend'] += 0.3
        df['Trend'] = df['Trend'].clip(-1.0, 1.0)
        
        df['Volume_MA20'] = df['Volume'].rolling(20).mean()
        df['Volatility'] = df['Close'].pct_change().rolling(20).std()
        
        return df
    
    def get_grid_spacing(self, atr: float, price: float, force_update: bool = False) -> float:
        """基于波动率聚类的动态网格间距"""
        if force_update or self._current_idx % 5 == 0:
            self.vol_cluster.detect_regime(atr, price, self._current_idx)
            
        return self.vol_cluster.get_dynamic_spacing(
            self.config.base_grid_pct, atr, price
        )
    
    def execute_buy_kelly(self, grid: Dict, price: float, date: pd.Timestamp,
                         total_value: float, is_trend_add: bool = False) -> bool:
        """使用Kelly准则的买入执行"""
        # 获取多时间框架信号
        can_trade, strength, alignment = self.mtf_momentum.should_trade(
            self._current_idx, 'long'
        ) if self.mtf_momentum else (True, 0.5, 0)
        
        if not can_trade and not is_trend_add:
            return False
            
        # 获取Kelly仓位
        vol_regime = self.vol_cluster.regime
        trend_score = self._df['Trend'].iloc[self._current_idx]
        
        shares, kelly_pct = self.kelly_sizer.get_position_size(
            total_value, price, vol_regime, trend_score, alignment
        )
        
        if shares <= 0 or kelly_pct <= 0:
            return False
            
        # 检查是否超过最大持仓
        current_pct = self.position * price / total_value
        if current_pct + kelly_pct > self.config.max_position:
            # 按比例缩减
            available_pct = self.config.max_position - current_pct
            scale = available_pct / kelly_pct if kelly_pct > 0 else 0
            shares *= scale
            kelly_pct *= scale
            
        cost = shares * price
        
        # 现金检查（保留储备金）
        cash_reserve = total_value * self.config.min_cash_reserve
        if cost > self.cash - cash_reserve:
            max_cost = self.cash - cash_reserve
            shares = max_cost / price if price > 0 else 0
            cost = shares * price
            
        if shares <= 0 or cost <= 0:
            return False
            
        self.cash -= cost
        self.position += shares
        
        # 更新成本基础
        if self.position > 0:
            total_cost = self.cost_basis * (self.position - shares) + cost
            self.cost_basis = total_cost / self.position
        else:
            self.cost_basis = price
            
        self._entry_max_price = max(self._entry_max_price, price)
        
        # 更新网格状态
        grid['active'] = True
        grid['shares'] = shares
        grid['entry_price'] = price
        grid['entry_date'] = date
        grid['kelly_pct'] = kelly_pct
        
        trade_type = 'TREND_ADD' if is_trend_add else 'BUY_KELLY'
        self.trades.append(Trade(
            date=date, type=trade_type, price=price,
            shares=shares, value=cost, grid_level=grid['level']
        ))
        
        return True
    
    def execute_sell(self, grid: Dict, price: float, date: pd.Timestamp) -> Optional[Trade]:
        """执行卖出并更新Kelly"""
        if not grid['active'] or grid['shares'] <= 0:
            return None
            
        shares = grid['shares']
        entry_price = grid['entry_price'] if grid['entry_price'] > 0 else grid['buy_price']
        
        value = shares * price
        self.cash += value
        
        cost = shares * entry_price
        pnl = value - cost
        
        self.position -= shares
        if self.position <= 1e-10:
            self.position = 0
            self.cost_basis = 0
            self._entry_max_price = 0
            
        trade = Trade(
            date=date, type='SELL', price=price,
            shares=shares, value=value, pnl=pnl, grid_level=grid['level']
        )
        self.trades.append(trade)
        
        # 更新Kelly估计器
        holding_days = (date - grid['entry_date']).days if grid['entry_date'] else 1
        trend_score = self._df['Trend'].iloc[self._current_idx]
        self.kelly_sizer.update_trade(pnl, entry_price, price, holding_days, 
                                      trend_score, self._current_idx)
        
        grid['active'] = False
        grid['shares'] = 0
        grid['entry_price'] = 0
        grid['entry_date'] = None
        grid['kelly_pct'] = 0
        
        return trade
    
    def execute_stop_loss(self, price: float, date: pd.Timestamp, 
                         stop_type: str = 'STOP_LOSS') -> Trade:
        """执行止损清仓"""
        if self.position <= 0:
            raise ValueError("持仓为空时不能止损")
            
        value = self.position * price
        self.cash += value
        
        trade = Trade(
            date=date, type=stop_type, price=price,
            shares=self.position, value=value
        )
        self.trades.append(trade)
        
        # 重置状态
        self.position = 0
        self.cost_basis = 0
        self._entry_max_price = 0
        
        # 初始化渐进式重启
        self.cooldown_counter = self.config.cooldown_days
        self._restart_phase = 3
        self._restart_base_cash = self.cash * 0.20  # 首批20%
        self._last_restart_idx = self._current_idx
        
        # 重置Kelly（保守模式）
        self.kelly_sizer.current_kelly *= 0.7
        
        if self.grid_manager:
            for grid in self.grid_manager.grids:
                if grid['active']:
                    grid['active'] = False
                    grid['shares'] = 0
                    
        return trade
    
    def check_stop_loss(self, total_value: float, price: float, date: pd.Timestamp) -> bool:
        """检查固定比例止损"""
        self.max_value = max(self.max_value, total_value)
        
        if self.max_value <= 0:
            return False
            
        drawdown = (self.max_value - total_value) / self.max_value
        
        if drawdown > self.config.stop_loss and self.position > 0:
            self.execute_stop_loss(price, date, 'STOP_LOSS')
            return True
            
        return False
    
    def check_atr_stop_loss(self, price: float, date: pd.Timestamp) -> bool:
        """ATR动态止损"""
        if self.position <= 0 or self._entry_max_price <= 0:
            return False
            
        atr = self._df['ATR'].iloc[self._current_idx]
        stop_price = self._entry_max_price - self.config.atr_stop_multiplier * atr
        
        if price < stop_price:
            self.execute_stop_loss(price, date, 'ATR_STOP')
            return True
            
        return False
    
    def should_update_grids(self, current_price: float) -> bool:
        """判断是否需要更新网格中心"""
        if self.grid_manager is None:
            return True
            
        last_center = self.grid_manager.center_price
        deviation = abs(current_price - last_center) / last_center if last_center > 0 else 0
        
        return deviation > self.config.position_update_threshold
    
    def should_trend_add(self, i: int, price: float, current_trend: float) -> bool:
        """多因子趋势加仓确认（增强版）"""
        if current_trend <= 0.5 or self.position <= 0:
            return False
            
        if (i - self._last_add_idx) < 8:
            return False
            
        # 多时间框架确认
        can_trade, strength, alignment = self.mtf_momentum.should_trade(i, 'long')
        if not can_trade or strength < 0.8:
            return False
            
        # 周线趋势过滤
        weekly_trend = self._df['Weekly_Trend'].iloc[i]
        if weekly_trend < 0:
            return False
            
        # 价格突破确认
        recent_high = self._df['High'].iloc[max(0, i-15):i].max()
        if price < recent_high * 0.985:
            return False
            
        # 成交量确认
        current_vol = self._df['Volume'].iloc[i]
        avg_vol = self._df['Volume_MA20'].iloc[i]
        if current_vol < avg_vol * 1.0:
            return False
            
        # RSI过滤
        if self._df['RSI'].iloc[i] > 78:
            return False
            
        # 波动率过滤（避免高波动聚类期间加仓）
        if self.vol_cluster.regime == 'high_clustering':
            return False
            
        return True
    
    def adaptive_rebalance(self, i: int, price: float):
        """智能再平衡"""
        should_rebal, reason = self.rebalance_trigger.check_rebalance(
            i, price, self.grid_manager
        )
        
        if not should_rebal:
            return
            
        if self.grid_manager is None:
            return
            
        # 根据原因调整策略
        new_spacing = self.get_grid_spacing(
            self._df['ATR'].iloc[i], price, force_update=True
        )
        
        if reason == "price_deviation":
            # 价格偏离：调整中心但保留盈利网格
            closed = self.grid_manager.update_center(price, new_spacing, preserve_active=True)
        elif reason == "regime_shift":
            # 状态转变：重置网格
            closed = self.grid_manager.update_center(price, new_spacing, preserve_active=False)
            # 使用Kelly重新建仓
            if self.cash > self.config.initial_capital * 0.3:
                total_value = self.cash + self.position * price
                inactive = self.grid_manager.get_inactive_buy_grids()
                if inactive:
                    self.execute_buy_kelly(inactive[0], price, 
                                          self._df['datetime'].iloc[i], total_value)
        else:
            closed = self.grid_manager.update_center(price, new_spacing, preserve_active=True)
            
        # 处理被迫平仓的网格
        for grid in closed:
            if grid['active'] and grid['shares'] > 0:
                sell_value = grid['shares'] * price
                self.cash += sell_value
                self.position -= grid['shares']
                pnl = sell_value - grid['shares'] * grid['entry_price']
                self.trades.append(Trade(
                    date=self._df['datetime'].iloc[i], type='GRID_CLOSE', 
                    price=price, shares=grid['shares'], 
                    value=sell_value, pnl=pnl
                ))
                # 更新Kelly
                self.kelly_sizer.update_trade(
                    pnl, grid['entry_price'], price,
                    (self._df['datetime'].iloc[i] - grid['entry_date']).days if grid['entry_date'] else 1,
                    self._df['Trend'].iloc[i], i
                )
                
        self.last_grid_update_idx = i
    
    def execute_restart_build(self, price: float, date: pd.Timestamp, total_value: float):
        """执行渐进式建仓（Kelly优化版）"""
        if self._restart_phase <= 0:
            return
            
        if (self._current_idx - self._last_restart_idx) < 4:
            return
            
        spacing = self.get_grid_spacing(self._df['ATR'].iloc[self._current_idx], price)
        # 根据阶段调整
        phase_multipliers = {3: 1.5, 2: 1.2, 1: 1.0}
        wide_spacing = spacing * phase_multipliers.get(self._restart_phase, 1.0)
        levels = max(3, self.config.grid_levels - (3 - self._restart_phase))
        
        self.grid_manager = GridManager(price, levels, wide_spacing)
        
        # 使用Kelly仓位而非固定比例
        vol_regime = self.vol_cluster.regime
        trend_score = self._df['Trend'].iloc[self._current_idx]
        
        # 重启时保守一些
        _, kelly_pct = self.kelly_sizer.get_position_size(
            total_value, price, vol_regime, trend_score * 0.8, 0.5
        )
        
        invest = total_value * kelly_pct * 0.5  # 重启时减半
        
        if self._restart_phase == 3:
            invest = min(invest, self.cash * 0.20)  # 首批最多20%
        elif self._restart_phase == 2:
            invest = min(invest, self.cash * 0.25)  # 第二批最多25%
        else:
            invest = min(invest, self.cash * 0.30)  # 最后一批最多30%
            
        shares = invest / price if price > 0 else 0
        
        if shares > 0 and invest <= self.cash * 0.95:
            self.cash -= invest
            self.position += shares
            self.cost_basis = price
            self._entry_max_price = price
            
            self.trades.append(Trade(
                date=date, type='RESTART_BUY', price=price,
                shares=shares, value=invest
            ))
            
        self._restart_phase -= 1
        self._last_restart_idx = self._current_idx
        self.last_grid_update_idx = self._current_idx
    
    def run_backtest(self, df: pd.DataFrame) -> pd.DataFrame:
        """运行回测 - v5.0主循环"""
        self.reset()
        
        print("预计算技术指标...")
        self._df = self.prepare_indicators(df)
        
        # 初始化多时间框架分析
        self.mtf_momentum = MultiTimeframeMomentum(self._df, self.config)
        
        start_idx = max(self.config.trend_period, self.config.vol_period, 60)
        n_bars = len(self._df)
        
        if n_bars <= start_idx:
            raise ValueError(f"数据长度不足，需要至少{start_idx}条数据")
            
        # 初始化网格
        init_price = self._df['Close'].iloc[start_idx]
        init_atr = self._df['ATR'].iloc[start_idx]
        spacing = self.get_grid_spacing(init_atr, init_price, force_update=True)
        self.grid_manager = GridManager(init_price, self.config.grid_levels, spacing)
        self.last_grid_update_idx = start_idx
        self._entry_max_price = init_price
        
        print(f"开始回测: {self._df['datetime'].iloc[start_idx]} 至 {self._df['datetime'].iloc[-1]}")
        print(f"初始Kelly仓位: {self.kelly_sizer.current_kelly:.2%}")
        
        for i in range(start_idx, n_bars):
            self._current_idx = i
            
            date = self._df['datetime'].iloc[i]
            price = self._df['Close'].iloc[i]
            high = self._df['High'].iloc[i]
            low = self._df['Low'].iloc[i]
            
            # 处理冷却期
            if self.cooldown_counter > 0:
                self.cooldown_counter -= 1
                total = self.cash + self.position * price
                self.max_value = max(self.max_value, total)
                
                # 记录状态
                mtf_data = self.mtf_momentum.get_composite_signal(i) if self.mtf_momentum else {}
                self.values.append({
                    'date': date, 'price': price, 'cash': self.cash,
                    'position': self.position, 'value': total,
                    'trend': self._df['Trend'].iloc[i],
                    'weekly_trend': self._df['Weekly_Trend'].iloc[i],
                    'kelly': self.kelly_sizer.current_kelly,
                    'vol_regime': self.vol_cluster.regime,
                    'mtf_alignment': mtf_data.get('alignment', 0)
                })
                
                # 渐进式重启建仓
                if self.cooldown_counter <= self.config.cooldown_days - 3:
                    self.execute_restart_build(price, date, total)
                continue
            
            # 更新波动率聚类状态
            self.vol_cluster.detect_regime(self._df['ATR'].iloc[i], price, i)
            
            # 智能再平衡检查
            self.adaptive_rebalance(i, price)
            
            # 定期更新网格中心
            current_trend = self._df['Trend'].iloc[i]
            if (i - self.last_grid_update_idx >= 5 or abs(current_trend) > 0.5):
                if self.should_update_grids(price):
                    if abs(current_trend) > 0.3 and self.grid_manager:
                        old_center = self.grid_manager.center_price
                        if current_trend > 0:
                            new_center = max(old_center, price * 0.96)
                        else:
                            new_center = min(old_center, price * 1.04)
                            
                        spacing = self.get_grid_spacing(self._df['ATR'].iloc[i], new_center)
                        closed = self.grid_manager.update_center(new_center, spacing, preserve_active=True)
                        
                        # 处理被迫平仓的网格
                        for grid in closed:
                            if grid['active'] and grid['shares'] > 0:
                                sell_value = grid['shares'] * price
                                self.cash += sell_value
                                self.position -= grid['shares']
                                pnl = sell_value - grid['shares'] * grid['entry_price']
                                self.trades.append(Trade(
                                    date=date, type='GRID_CLOSE', price=price,
                                    shares=grid['shares'], value=sell_value, pnl=pnl
                                ))
                                # 更新Kelly
                                self.kelly_sizer.update_trade(
                                    pnl, grid['entry_price'], price,
                                    (date - grid['entry_date']).days if grid['entry_date'] else 1,
                                    current_trend, i
                                )
                                
                    self.last_grid_update_idx = i
            
            # 计算市值
            position_value = self.position * price
            total_value = self.cash + position_value
            
            # 双重止损检查
            if self.check_stop_loss(total_value, price, date):
                mtf_data = self.mtf_momentum.get_composite_signal(i) if self.mtf_momentum else {}
                self.values.append({
                    'date': date, 'price': price, 'cash': self.cash,
                    'position': 0, 'value': total_value,
                    'trend': current_trend, 'weekly_trend': self._df['Weekly_Trend'].iloc[i],
                    'kelly': self.kelly_sizer.current_kelly,
                    'vol_regime': self.vol_cluster.regime,
                    'mtf_alignment': mtf_data.get('alignment', 0)
                })
                continue
                
            if self.check_atr_stop_loss(price, date):
                mtf_data = self.mtf_momentum.get_composite_signal(i) if self.mtf_momentum else {}
                self.values.append({
                    'date': date, 'price': price, 'cash': self.cash,
                    'position': 0, 'value': total_value,
                    'trend': current_trend, 'weekly_trend': self._df['Weekly_Trend'].iloc[i],
                    'kelly': self.kelly_sizer.current_kelly,
                    'vol_regime': self.vol_cluster.regime,
                    'mtf_alignment': mtf_data.get('alignment', 0)
                })
                continue
            
            # 成交量过滤（放宽）
            vol_ok = True
            if i >= 20:
                avg_vol = self._df['Volume_MA20'].iloc[i]
                vol_ok = self._df['Volume'].iloc[i] > avg_vol * 0.5
            
            # 网格交易（使用Kelly仓位）
            if self.grid_manager and vol_ok and current_trend > -0.6:
                buy_signals, sell_signals = self.grid_manager.check_signals(high, low)
                
                # 先处理卖出
                for grid in sell_signals:
                    self.execute_sell(grid, price, date)
                    
                # 再处理买入（Kelly仓位）
                for grid in buy_signals:
                    self.execute_buy_kelly(grid, price, date, total_value)
            
            # 趋势加仓（多因子确认）
            if self.should_trend_add(i, price, current_trend):
                current_pct = self.position * price / total_value
                if current_pct < self.config.max_position * 0.85:
                    inactive = self.grid_manager.get_inactive_buy_grids()
                    if inactive:
                        if self.execute_buy_kelly(inactive[0], price, date, total_value, 
                                                  is_trend_add=True):
                            self._last_add_idx = i
            
            # 趋势减仓（多时间框架确认）
            if current_trend < -0.7 and self.position > 0:
                weekly_trend = self._df['Weekly_Trend'].iloc[i]
                mtf_data = self.mtf_momentum.get_composite_signal(i) if self.mtf_momentum else {}
                
                # 需要多时间框架确认
                if weekly_trend < 0 and mtf_data.get('alignment', 0) < -0.3:
                    reduce_pct = min(0.35, abs(current_trend))
                    reduce_shares = self.position * reduce_pct
                    
                    if reduce_shares > 0:
                        value = reduce_shares * price
                        self.cash += value
                        self.position -= reduce_shares
                        
                        self.trades.append(Trade(
                            date=date, type='TREND_REDUCE', price=price,
                            shares=reduce_shares, value=value
                        ))
            
            # 记录每日数据
            total_value = self.cash + self.position * price
            mtf_data = self.mtf_momentum.get_composite_signal(i) if self.mtf_momentum else {}
            self.values.append({
                'date': date, 'price': price, 'cash': self.cash,
                'position': self.position, 'value': total_value,
                'trend': current_trend, 'weekly_trend': self._df['Weekly_Trend'].iloc[i],
                'kelly': self.kelly_sizer.current_kelly,
                'vol_regime': self.vol_cluster.regime,
                'mtf_alignment': mtf_data.get('alignment', 0)
            })
        
        results = pd.DataFrame(self.values)
        print(f"回测完成: 共{len(results)}个交易日，{len(self.trades)}笔交易")
        print(f"最终Kelly仓位: {self.kelly_sizer.current_kelly:.2%}")
        return results
    
    def get_performance_metrics(self, results: pd.DataFrame) -> Dict:
        """计算全面绩效指标（增强版）"""
        if len(results) == 0:
            return {}
            
        initial = self.config.initial_capital
        final = results['value'].iloc[-1]
        total_return = (final - initial) / initial if initial > 0 else 0
        
        days = (results['date'].iloc[-1] - results['date'].iloc[0]).days
        years = days / 365.25 if days > 0 else 0
        annual_return = (1 + total_return) ** (1/years) - 1 if years > 0 else 0
        
        # 最大回撤
        cummax = results['value'].cummax()
        drawdowns = (cummax - results['value']) / cummax
        max_dd = drawdowns.max() if len(cummax) > 0 else 0
        
        # 收益率序列
        daily_ret = results['value'].pct_change().dropna()
        
        # 夏普比率
        if len(daily_ret) > 1 and daily_ret.std() > 0:
            sharpe = ((daily_ret.mean() * 252 - 0.02) / 
                     (daily_ret.std() * np.sqrt(252)))
        else:
            sharpe = 0
            
        # Sortino Ratio
        downside_ret = daily_ret[daily_ret < 0]
        downside_std = downside_ret.std() * np.sqrt(252) if len(downside_ret) > 0 else 0
        sortino = ((daily_ret.mean() * 252 - 0.02) / downside_std) if downside_std > 0 else 0
        
        # 新增：Kelly效率指标
        avg_kelly = results['kelly'].mean() if 'kelly' in results.columns else 0
        kelly_utilization = avg_kelly / self.config.kelly_max_position if self.config.kelly_max_position > 0 else 0
        
        # 新增：多时间框架一致性胜率
        if 'mtf_alignment' in results.columns:
            high_alignment_days = (abs(results['mtf_alignment']) > 0.5).sum()
            mtf_win_rate = high_alignment_days / len(results) if len(results) > 0 else 0
        else:
            mtf_win_rate = 0
        
        # 交易统计
        trades_df = pd.DataFrame([{
            'type': t.type, 'pnl': t.pnl, 'value': t.value, 'date': t.date
        } for t in self.trades])
        
        metrics = {
            'initial_capital': initial,
            'final_value': final,
            'total_return_pct': total_return * 100,
            'annual_return_pct': annual_return * 100,
            'max_drawdown_pct': max_dd * 100,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'calmar_ratio': annual_return / max_dd if max_dd > 0 else 0,
            'total_trades': len(self.trades),
            'avg_kelly': avg_kelly,
            'kelly_utilization': kelly_utilization,
            'mtf_consistency_rate': mtf_win_rate
        }
        
        if len(trades_df) > 0:
            type_counts = trades_df['type'].value_counts().to_dict()
            metrics.update({
                'buy_trades': type_counts.get('BUY_KELLY', 0),
                'sell_trades': type_counts.get('SELL', 0),
                'stop_loss_trades': type_counts.get('STOP_LOSS', 0) + type_counts.get('ATR_STOP', 0),
                'trend_add_trades': type_counts.get('TREND_ADD', 0),
                'trend_reduce_trades': type_counts.get('TREND_REDUCE', 0),
                'grid_close_trades': type_counts.get('GRID_CLOSE', 0),
                'restart_trades': type_counts.get('RESTART_BUY', 0),
            })
            
            # 盈亏分析
            sell_trades = trades_df[trades_df['type'].isin(['SELL', 'GRID_CLOSE'])]
            if len(sell_trades) > 0:
                pnls = sell_trades['pnl'].dropna()
                wins = pnls[pnls > 0]
                losses = pnls[pnls < 0]
                
                metrics['win_rate_pct'] = len(wins) / len(pnls) * 100 if len(pnls) > 0 else 0
                metrics['avg_pnl'] = pnls.mean()
                metrics['total_pnl'] = pnls.sum()
                metrics['avg_win'] = wins.mean() if len(wins) > 0 else 0
                metrics['avg_loss'] = losses.mean() if len(losses) > 0 else 0
                metrics['profit_factor'] = abs(wins.sum() / losses.sum()) if len(losses) > 0 and losses.sum() != 0 else float('inf')
                metrics['win_loss_ratio'] = abs(metrics['avg_win'] / metrics['avg_loss']) if metrics['avg_loss'] != 0 else 0
                
                # Kelly策略特有：仓位效率
                kelly_trades = trades_df[trades_df['type'] == 'BUY_KELLY']
                if len(kelly_trades) > 0:
                    metrics['kelly_trade_ratio'] = len(kelly_trades) / len(trades_df) * 100
        
        return metrics


class StrategyVisualizerV5:
    """策略可视化 v5.0 - 新增Kelly和MTF面板"""
    
    def __init__(self, config: StrategyConfig):
        self.config = config
        self.colors = {
            'strategy': '#1f77b4', 'buyhold': '#ff7f0e', 'profit': '#2ca02c',
            'loss': '#d62728', 'buy': '#00aa00', 'sell': '#00ff00',
            'stop': '#ff0000', 'atr_stop': '#ff4444', 'add': '#0000ff',
            'reduce': '#ff8800', 'grid_close': '#ff00ff', 'restart': '#00ffff',
            'kelly': '#9467bd', 'mtf': '#8c564b'
        }
    
    def align_data(self, results: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """使用merge_asof进行时间对齐"""
        results = results.copy()
        df_aligned = pd.merge_asof(
            results.sort_values('date'),
            df[['datetime', 'Close']].rename(columns={'datetime': 'date', 'Close': 'buyhold_price'}).sort_values('date'),
            on='date',
            direction='backward'
        )
        return df_aligned
    
    def create_comprehensive_report(self, df: pd.DataFrame, 
                                   results: pd.DataFrame,
                                   trades: List[Trade],
                                   output_path: str):
        """生成综合分析报告 v5.0"""
        trades_df = pd.DataFrame([{
            'date': t.date, 'type': t.type, 'price': t.price,
            'shares': t.shares, 'value': t.value, 'pnl': t.pnl
        } for t in trades])
        
        aligned_results = self.align_data(results, df)
        
        fig = plt.figure(figsize=(22, 32))
        gs = GridSpec(10, 2, height_ratios=[3, 2, 2, 2, 2, 2, 2, 2, 2, 2], 
                     hspace=0.35, wspace=0.25)
        
        # 1. 主图：净值对比（保持不变）
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(results['date'], results['value'], 
                label='Kelly Grid Strategy v5.0', linewidth=2.5, 
                color=self.colors['strategy'], alpha=0.9)
        
        if 'buyhold_price' in aligned_results.columns and aligned_results['buyhold_price'].notna().any():
            first_valid = aligned_results['buyhold_price'].first_valid_index()
            if first_valid is not None:
                initial_price = aligned_results.loc[first_valid, 'buyhold_price']
                buyhold_values = aligned_results['buyhold_price'] / initial_price * self.config.initial_capital
                ax1.plot(aligned_results['date'], buyhold_values, 
                        label='Buy & Hold', linewidth=2, 
                        color=self.colors['buyhold'], alpha=0.7, linestyle='--')
                buyhold_final = buyhold_values.iloc[-1]
        else:
            buyhold_final = self.config.initial_capital
        
        ax1.axhline(y=self.config.initial_capital, color='gray', linestyle=':', alpha=0.5)
        ax1.fill_between(results['date'], self.config.initial_capital, results['value'],
                        where=(results['value'] >= self.config.initial_capital), 
                        alpha=0.2, color=self.colors['profit'])
        ax1.fill_between(results['date'], self.config.initial_capital, results['value'],
                        where=(results['value'] < self.config.initial_capital), 
                        alpha=0.2, color=self.colors['loss'])
        
        final_value = results['value'].iloc[-1]
        strategy_return = (final_value / self.config.initial_capital - 1) * 100
        buyhold_return = (buyhold_final / self.config.initial_capital - 1) * 100
        
        ax1.set_title(f'Trend-Enhanced Kelly Grid Strategy v5.0\n' + 
                     f'Strategy: {strategy_return:.1f}% vs Buy&Hold: {buyhold_return:.1f}%',
                     fontsize=16, fontweight='bold', pad=20)
        ax1.set_ylabel('Portfolio Value (CNY)', fontsize=12)
        ax1.legend(loc='upper left', fontsize=11)
        ax1.grid(True, alpha=0.3, linestyle='--')
        
        # 2. 价格与交易点
        ax2 = fig.add_subplot(gs[1, :])
        ax2.plot(df['datetime'], df['Close'], label='Stock Price', 
                color='black', linewidth=1.5, alpha=0.8)
        
        if len(trades_df) > 0:
            for trade_type, marker, color_key in [
                ('BUY_KELLY', '^', 'buy'), ('SELL', 'v', 'sell'),
                ('STOP_LOSS', 'x', 'stop'), ('ATR_STOP', 'X', 'atr_stop'),
                ('TREND_ADD', '*', 'add'), ('TREND_REDUCE', 'D', 'reduce'),
                ('GRID_CLOSE', 'p', 'grid_close'), ('RESTART_BUY', 'h', 'restart')
            ]:
                subset = trades_df[trades_df['type'] == trade_type]
                if len(subset) > 0:
                    ax2.scatter(subset['date'], subset['price'], 
                               marker=marker, s=120, color=self.colors[color_key],
                               alpha=0.8, label=f'{trade_type} ({len(subset)})', 
                               zorder=5, edgecolors='black', linewidth=0.5)
        
        ax2.set_title('Stock Price & Trading Signals (Kelly Optimized)', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Price (CNY)', fontsize=11)
        ax2.legend(loc='upper left', ncol=4, fontsize=9)
        ax2.grid(True, alpha=0.3)
        
        # 3. 持仓比例 + Kelly仓位线
        ax3 = fig.add_subplot(gs[2, 0])
        position_pct = results['position'] * results['price'] / results['value'] * 100
        ax3.fill_between(results['date'], 0, position_pct, alpha=0.6, color='steelblue')
        ax3.plot(results['date'], position_pct, color='navy', linewidth=1.5, label='Actual Position')
        
        # 添加Kelly建议仓位线
        if 'kelly' in results.columns:
            kelly_line = results['kelly'] * 100
            ax3.plot(results['date'], kelly_line, color=self.colors['kelly'], 
                    linewidth=1.5, linestyle='--', label='Kelly Optimal %', alpha=0.8)
        
        ax3.axhline(y=self.config.max_position*100, color='darkred', 
                   linestyle='--', alpha=0.3, label=f'Max {self.config.max_position*100:.0f}%')
        ax3.set_ylabel('Position Ratio (%)', fontsize=11)
        ax3.set_title('Position Allocation vs Kelly Recommendation', fontsize=13, fontweight='bold')
        ax3.legend(loc='upper right', fontsize=9)
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 100)
        
        # 4. 现金余额
        ax4 = fig.add_subplot(gs[2, 1])
        cash_reserve_line = results['value'] * self.config.min_cash_reserve
        ax4.fill_between(results['date'], 0, results['cash'], 
                        alpha=0.5, color='green', label='Cash')
        ax4.plot(results['date'], results['cash'], color='darkgreen', linewidth=1.5)
        ax4.plot(results['date'], cash_reserve_line, color='red', linestyle='--', alpha=0.5, label='Reserve Line')
        ax4.set_ylabel('Cash (CNY)', fontsize=11)
        ax4.set_title('Cash Reserve', fontsize=13, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. 多时间框架一致性
        ax5 = fig.add_subplot(gs[3, 0])
        if 'mtf_alignment' in results.columns:
            alignment = results['mtf_alignment']
            ax5.fill_between(results['date'], -1, 1, where=(alignment > 0.3), 
                            alpha=0.3, color='green', label='Bullish Consensus')
            ax5.fill_between(results['date'], -1, 1, where=(alignment < -0.3), 
                            alpha=0.3, color='red', label='Bearish Consensus')
            ax5.plot(results['date'], alignment, color=self.colors['mtf'], linewidth=1.5)
            ax5.axhline(y=0.33, color='green', linestyle='--', alpha=0.5)
            ax5.axhline(y=-0.33, color='red', linestyle='--', alpha=0.5)
        ax5.set_ylabel('MTF Alignment', fontsize=11)
        ax5.set_title('Multi-Timeframe Consensus (-1 to 1)', fontsize=13, fontweight='bold')
        ax5.set_ylim(-1.2, 1.2)
        ax5.grid(True, alpha=0.3)
        
        # 6. 波动率聚类状态
        ax6 = fig.add_subplot(gs[3, 1])
        if 'vol_regime' in results.columns:
            regime_map = {'low': 1, 'low_clustering': 2, 'normal': 3, 'high': 4, 'high_clustering': 5}
            regime_numeric = results['vol_regime'].map(regime_map).fillna(3)
            colors_regime = ['lightgreen', 'green', 'yellow', 'orange', 'red']
            for i, (regime, val) in enumerate(regime_map.items()):
                mask = results['vol_regime'] == regime
                if mask.any():
                    ax6.scatter(results['date'][mask], regime_numeric[mask], 
                               c=self.colors.get(regime, 'gray'), label=regime, s=10, alpha=0.7)
        ax6.set_ylabel('Volatility Regime', fontsize=11)
        ax6.set_title('Volatility Clustering Detection', fontsize=13, fontweight='bold')
        ax6.set_yticks([1, 2, 3, 4, 5])
        ax6.set_yticklabels(['Low', 'LowClust', 'Normal', 'High', 'HighClust'])
        ax6.grid(True, alpha=0.3)
        
        # 7. 回撤
        ax7 = fig.add_subplot(gs[4, 0])
        cummax = results['value'].cummax()
        drawdown = (cummax - results['value']) / cummax * 100
        ax7.fill_between(results['date'], 0, drawdown, alpha=0.5, color='red')
        ax7.plot(results['date'], drawdown, color='darkred', linewidth=1.5)
        max_dd = drawdown.max()
        max_dd_idx = drawdown.idxmax()
        ax7.scatter(results['date'].iloc[max_dd_idx], max_dd, s=100, color='darkred', zorder=5)
        ax7.annotate(f'Max DD: {max_dd:.1f}%', 
                    xy=(results['date'].iloc[max_dd_idx], max_dd),
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8),
                    fontsize=10, fontweight='bold')
        ax7.set_ylabel('Drawdown (%)', fontsize=11)
        ax7.set_title('Drawdown Analysis', fontsize=13, fontweight='bold')
        ax7.grid(True, alpha=0.3)
        
        # 8. RSI
        ax8 = fig.add_subplot(gs[4, 1])
        ax8.plot(df['datetime'], df['RSI'], color='purple', linewidth=1)
        ax8.axhline(y=70, color='red', linestyle='--', alpha=0.5)
        ax8.axhline(y=30, color='green', linestyle='--', alpha=0.5)
        ax8.fill_between(df['datetime'], 30, 70, alpha=0.1, color='gray')
        ax8.set_ylabel('RSI', fontsize=11)
        ax8.set_title('RSI Indicator', fontsize=13, fontweight='bold')
        ax8.set_ylim(0, 100)
        ax8.grid(True, alpha=0.3)
        
        # 9. 盈亏分布
        ax9 = fig.add_subplot(gs[5, 0])
        sell_trades = trades_df[trades_df['type'].isin(['SELL', 'GRID_CLOSE'])]
        if len(sell_trades) > 0 and 'pnl' in sell_trades.columns:
            pnls = sell_trades['pnl'].dropna()
            if len(pnls) > 0:
                colors_pnl = ['green' if p > 0 else 'red' for p in pnls]
                ax9.bar(range(len(pnls)), pnls, color=colors_pnl, alpha=0.7)
                ax9.axhline(y=0, color='black', linestyle='-', linewidth=1)
                win_rate = (pnls > 0).sum() / len(pnls) * 100
                ax9.set_xlabel('Trade Number', fontsize=11)
                ax9.set_ylabel('PnL (CNY)', fontsize=11)
                ax9.set_title(f'PnL Distribution (Win Rate: {win_rate:.1f}%)', fontsize=13, fontweight='bold')
                ax9.grid(True, alpha=0.3, axis='y')
        
        # 10. 交易统计
        ax10 = fig.add_subplot(gs[5, 1])
        trade_types = ['BUY_KELLY', 'SELL', 'STOP_LOSS', 'ATR_STOP', 'TREND_ADD', 
                      'TREND_REDUCE', 'GRID_CLOSE', 'RESTART_BUY']
        trade_counts = [len(trades_df[trades_df['type']==t]) for t in trade_types]
        colors_type = ['green', 'lime', 'red', 'darkred', 'blue', 'orange', 'magenta', 'cyan']
        bars = ax10.bar([t.replace('_', '\n') for t in trade_types], trade_counts, 
                       color=colors_type, alpha=0.7)
        ax10.set_ylabel('Count', fontsize=11)
        ax10.set_title('Trade Statistics by Type', fontsize=13, fontweight='bold')
        for bar, count in zip(bars, trade_counts):
            if count > 0:
                ax10.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(trade_counts)*0.01, 
                         str(count), ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax10.grid(True, alpha=0.3, axis='y')
        
        # 11. 滚动夏普比率
        ax11 = fig.add_subplot(gs[6, 0])
        rolling_returns = results['value'].pct_change()
        rolling_sharpe = (rolling_returns.rolling(60).mean() * 252 - 0.02) / (rolling_returns.rolling(60).std() * np.sqrt(252))
        ax11.plot(results['date'], rolling_sharpe, color='blue', linewidth=1)
        ax11.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax11.axhline(y=1, color='green', linestyle='--', alpha=0.3)
        ax11.set_ylabel('Sharpe Ratio', fontsize=11)
        ax11.set_title('Rolling Sharpe Ratio (60-day)', fontsize=13, fontweight='bold')
        ax11.grid(True, alpha=0.3)
        
        # 12. 月度收益热力图
        ax12 = fig.add_subplot(gs[6, 1])
        results['month'] = results['date'].dt.to_period('M')
        monthly_returns = results.groupby('month')['value'].apply(lambda x: (x.iloc[-1] / x.iloc[0] - 1) * 100)
        if len(monthly_returns) > 0:
            monthly_df = monthly_returns.reset_index()
            monthly_df['year'] = monthly_df['month'].dt.year
            monthly_df['mon'] = monthly_df['month'].dt.month
            pivot_table = monthly_df.pivot(index='year', columns='mon', values='value')
            im = ax12.imshow(pivot_table.values, cmap='RdYlGn', aspect='auto', vmin=-15, vmax=15)
            ax12.set_xticks(range(12))
            ax12.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
            ax12.set_yticks(range(len(pivot_table)))
            ax12.set_yticklabels(pivot_table.index)
            ax12.set_title('Monthly Returns Heatmap (%)', fontsize=13, fontweight='bold')
            plt.colorbar(im, ax=ax12)
        
        # 13. ATR与网格间距
        ax13 = fig.add_subplot(gs[7, 0])
        ax13.plot(df['datetime'], df['ATR'] / df['Close'] * 100, 
                color='orange', linewidth=1, label='ATR %')
        ax13.axhline(y=self.config.base_grid_pct*100, color='blue', linestyle='--', alpha=0.5)
        ax13.set_ylabel('ATR / Price (%)', fontsize=11)
        ax13.set_title('Volatility vs Grid Spacing', fontsize=13, fontweight='bold')
        ax13.legend()
        ax13.grid(True, alpha=0.3)
        
        # 14. Kelly仓位历史
        ax14 = fig.add_subplot(gs[7, 1])
        if 'kelly' in results.columns:
            ax14.fill_between(results['date'], 0, results['kelly']*100, alpha=0.4, color=self.colors['kelly'])
            ax14.plot(results['date'], results['kelly']*100, color=self.colors['kelly'], linewidth=1.5)
            ax14.axhline(y=self.config.kelly_max_position*100, color='red', linestyle='--', alpha=0.5)
        ax14.set_ylabel('Kelly %', fontsize=11)
        ax14.set_title('Kelly Optimal Position Size Evolution', fontsize=13, fontweight='bold')
        ax14.grid(True, alpha=0.3)
        
        # 15. 资金曲线与回撤叠加
        ax15 = fig.add_subplot(gs[8, :])
        ax15_twin = ax15.twinx()
        ax15.plot(results['date'], results['value'], color='blue', linewidth=1.5, label='Portfolio Value')
        ax15_twin.fill_between(results['date'], 0, drawdown, alpha=0.3, color='red', label='Drawdown')
        ax15.set_ylabel('Value (CNY)', color='blue', fontsize=11)
        ax15_twin.set_ylabel('Drawdown (%)', color='red', fontsize=11)
        ax15.set_title('Equity Curve with Drawdown Overlay', fontsize=13, fontweight='bold')
        ax15.grid(True, alpha=0.3)
        
        # 16. 成交量
        ax16 = fig.add_subplot(gs[9, :])
        price_change = df['Close'].diff().fillna(0)
        colors_vol = ['red' if c >= 0 else 'green' for c in price_change]
        ax16.bar(df['datetime'], df['Volume']/1e6, color=colors_vol, alpha=0.6, width=1)
        ax16.set_ylabel('Volume (M)', fontsize=11)
        ax16.set_title('Trading Volume', fontsize=13, fontweight='bold')
        ax16.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_file = Path(output_path) / 'comprehensive_analysis_v5_kelly.png'
        plt.savefig(output_file, dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"✅ 综合分析图表已保存: {output_file}")
        return output_file


def print_performance_report_v5(metrics: Dict):
    """打印格式化绩效报告 v5.0"""
    print("\n" + "="*80)
    print(" " * 20 + "策略绩效报告 v5.0 - Kelly优化版")
    print("="*80)
    
    print(f"{'初始资金:':<30} {metrics.get('initial_capital', 0):>15,.0f} CNY")
    print(f"{'最终价值:':<30} {metrics.get('final_value', 0):>15,.0f} CNY")
    print(f"{'总收益率:':<30} {metrics.get('total_return_pct', 0):>15.2f} %")
    print(f"{'年化收益:':<30} {metrics.get('annual_return_pct', 0):>15.2f} %")
    print(f"{'最大回撤:':<30} {metrics.get('max_drawdown_pct', 0):>15.2f} %")
    print("-"*80)
    print(f"{'夏普比率:':<30} {metrics.get('sharpe_ratio', 0):>15.2f}")
    print(f"{'索提诺比率:':<30} {metrics.get('sortino_ratio', 0):>15.2f}")
    print(f"{'卡玛比率:':<30} {metrics.get('calmar_ratio', 0):>15.2f}")
    print(f"{'盈亏比:':<30} {metrics.get('win_loss_ratio', 0):>15.2f}")
    print(f"{'盈利因子:':<30} {metrics.get('profit_factor', 0):>15.2f}")
    print("-"*80)
    
    # Kelly特有指标
    print(f"{'平均Kelly仓位:':<30} {metrics.get('avg_kelly', 0)*100:>15.2f} %")
    print(f"{'Kelly利用率:':<30} {metrics.get('kelly_utilization', 0)*100:>15.2f} %")
    print(f"{'MTF一致性胜率:':<30} {metrics.get('mtf_consistency_rate', 0)*100:>15.2f} %")
    print("-"*80)
    
    print(f"{'总交易次数:':<30} {metrics.get('total_trades', 0):>15}")
    print(f"{'Kelly买入次数:':<30} {metrics.get('buy_trades', 0):>15}")
    print(f"{'趋势加仓次数:':<30} {metrics.get('trend_add_trades', 0):>15}")
    print(f"{'卖出次数:':<30} {metrics.get('sell_trades', 0):>15}")
    print(f"{'固定止损次数:':<30} {metrics.get('stop_loss_trades', 0):>15}")
    print(f"{'ATR止损次数:':<30} {metrics.get('atr_stop_trades', 0):>15}")
    print(f"{'网格强制平仓:':<30} {metrics.get('grid_close_trades', 0):>15}")
    print(f"{'趋势减仓:':<30} {metrics.get('trend_reduce_trades', 0):>15}")
    print(f"{'重启建仓:':<30} {metrics.get('restart_trades', 0):>15}")
    print("-"*80)
    print(f"{'胜率:':<30} {metrics.get('win_rate_pct', 0):>15.1f} %")
    print(f"{'平均盈亏:':<30} {metrics.get('avg_pnl', 0):>15,.0f} CNY")
    print(f"{'平均盈利:':<30} {metrics.get('avg_win', 0):>15,.0f} CNY")
    print(f"{'平均亏损:':<30} {metrics.get('avg_loss', 0):>15,.0f} CNY")
    print(f"{'总盈亏:':<30} {metrics.get('total_pnl', 0):>15,.0f} CNY")
    print("="*80)


def main():
    """主函数 - v5.0"""
    DATA_PATH = 'C:/Users/1/Desktop/python量化/603993历史数据(2020-2025).csv'
    OUTPUT_PATH = 'C:/Users/1/Desktop'
    
    print("="*80)
    print("趋势增强型智能网格策略 v5.0 - Kelly优化版")
    print("核心特性：Kelly动态仓位 + 波动率聚类 + 多时间框架共振")
    print("="*80)
    
    print("\n1. 加载数据...")
    try:
        df = pd.read_csv(DATA_PATH)
        print(f"   原始数据: {len(df)} 行")
    except Exception as e:
        print(f"   ❌ 数据加载失败: {e}")
        return
    
    print("\n2. 初始化策略...")
    config = StrategyConfig(
        initial_capital=100000,
        base_grid_pct=0.03,           # 降低基础间距
        grid_levels=7,                # 增加层级
        max_position=0.95,            # 提高上限
        stop_loss=0.15,               # 放宽止损
        cooldown_days=8,              # 缩短冷却期
        position_update_threshold=0.06,
        min_cash_reserve=0.05,        # 降低现金储备
        kelly_lookback=40,            # Kelly回望窗口
        kelly_fraction=0.25,          # 1/4 Kelly保守系数
        kelly_max_position=0.25       # 单标的上限
    )
    
    strategy = TrendEnhancedGridStrategyV5(config)
    
    print("\n3. 运行回测...")
    try:
        results = strategy.run_backtest(df)
    except Exception as e:
        print(f"   ❌ 回测失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n4. 计算绩效指标...")
    metrics = strategy.get_performance_metrics(results)
    print_performance_report_v5(metrics)
    
    print("\n5. 生成可视化图表...")
    try:
        visualizer = StrategyVisualizerV5(config)
        visualizer.create_comprehensive_report(
            strategy._df, results, strategy.trades, OUTPUT_PATH
        )
    except Exception as e:
        print(f"   ⚠️ 图表生成失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80)
    print("✅ 回测完成！")
    print(f"输出目录: {OUTPUT_PATH}")
    print("="*80)


if __name__ == "__main__":
    main()