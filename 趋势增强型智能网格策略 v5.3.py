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
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import seaborn as sns

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-whitegrid')


@dataclass(slots=True)
class Trade:
    date: pd.Timestamp
    type: str
    price: float
    shares: float
    value: float
    pnl: Optional[float] = None
    grid_level: Optional[int] = None


@dataclass
class StrategyConfig:
    """修复版配置参数 - 添加缺失属性"""
    initial_capital: float = 100000.0
    base_grid_pct: float = 0.025
    grid_levels: int = 10
    max_position: float = 0.95
    stop_loss: float = 0.20
    trend_period: int = 20
    vol_period: int = 14
    cooldown_days: int = 5
    min_grid_spacing: float = 0.01
    max_grid_spacing: float = 0.15
    position_update_threshold: float = 0.05
    min_cash_reserve: float = 0.02
    weekly_trend_period: int = 20
    atr_stop_multiplier: float = 3.0
    rebalance_interval: int = 15
    
    # Kelly参数
    kelly_lookback: int = 20
    kelly_fraction: float = 0.5
    kelly_max_position: float = 0.30
    kelly_initial: float = 0.15
    
    # 波动率聚类参数 - 添加缺失的属性
    vol_cluster_window: int = 20
    vol_cluster_threshold: float = 0.3
    
    # 多时间框架参数
    mtf_periods: Dict[str, int] = field(default_factory=lambda: {
        'short': 5, 'medium': 20, 'long': 60
    })
    
    # 交易成本
    commission_rate: float = 0.0003
    slippage: float = 0.0001


class KellyPositionSizer:
    """修复版Kelly仓位管理器"""
    
    def __init__(self, config: StrategyConfig):
        self.config = config
        self.trade_history: deque = deque(maxlen=config.kelly_lookback)
        self.current_kelly: float = config.kelly_initial
        self.win_streak: int = 0
        self.loss_streak: int = 0
        self.last_update_idx: int = 0
        self.initialized: bool = False
        
    def update_trade(self, pnl: float, entry_price: float, exit_price: float, 
                     holding_days: int, trend_score: float, idx: int):
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
        
        if pnl > 0:
            self.win_streak += 1
            self.loss_streak = 0
        else:
            self.loss_streak += 1
            self.win_streak = 0
        
        self._recalculate_kelly()
        self.initialized = True
    
    def _recalculate_kelly(self):
        n = len(self.trade_history)
        if n < 5:
            return
            
        returns = np.array([t['return'] for t in self.trade_history])
        
        wins = returns[returns > 0]
        losses = returns[returns < 0]
        
        W = len(wins) / n if n > 0 else 0.5
        
        if len(losses) > 0 and abs(np.mean(losses)) > 1e-6:
            R = np.mean(wins) / abs(np.mean(losses)) if len(wins) > 0 else 1.0
        else:
            R = 1.5
            
        if R > 0:
            kelly_raw = (W * R - (1 - W)) / R
            kelly_raw = np.clip(kelly_raw, -0.5, 1.0)
            kelly_adj = kelly_raw * self.config.kelly_fraction
            
            if self.win_streak >= 3:
                kelly_adj *= 1.1
            elif self.loss_streak >= 2:
                kelly_adj *= 0.9
                
            self.current_kelly = np.clip(kelly_adj, 0.05, self.config.kelly_max_position)
        
    def get_position_size(self, total_value: float, price: float, 
                         volatility_regime: str, trend_strength: float,
                         alignment_score: float) -> Tuple[float, float]:
        if total_value <= 0 or price <= 0:
            return 0.0, 0.0
            
        if not self.initialized and len(self.trade_history) < 5:
            base_size = self.config.kelly_initial
        else:
            base_size = max(self.current_kelly, 0.05)
        
        vol_multipliers = {
            'low': 1.2,
            'normal': 1.0,
            'high': 0.7,
            'high_clustering': 0.5
        }
        vol_mult = vol_multipliers.get(volatility_regime, 1.0)
        
        alignment_mult = 1 + abs(alignment_score) * 0.3
        trend_mult = 1 + max(0, trend_strength) * 0.2
        
        adjusted_pct = base_size * vol_mult * alignment_mult * trend_mult
        adjusted_pct = min(adjusted_pct, self.config.kelly_max_position)
        
        invest_amount = total_value * adjusted_pct * 0.98
        shares = invest_amount / price if price > 0 else 0
        
        return shares, adjusted_pct


class VolatilityClusteringGrid:
    """波动率聚类检测"""
    
    def __init__(self, config: StrategyConfig):
        self.config = config
        self.volatility_memory: deque = deque(maxlen=config.vol_cluster_window)
        self.regime: str = 'normal'
        self.clustering_factor: float = 1.0
        
    def detect_regime(self, current_atr: float, price: float, idx: int) -> str:
        if price <= 0 or current_atr <= 0:
            return self.regime
            
        vol_pct = current_atr / price
        self.volatility_memory.append(vol_pct)
        
        if len(self.volatility_memory) < 10:
            return 'normal'
            
        vol_series = pd.Series(self.volatility_memory)
        current_vol = vol_series.iloc[-1]
        mean_vol = vol_series.mean()
        
        if current_vol > mean_vol * 1.5:
            self.regime = 'high'
            self.clustering_factor = 1.3
        elif current_vol < mean_vol * 0.6:
            self.regime = 'low'
            self.clustering_factor = 0.8
        else:
            self.regime = 'normal'
            self.clustering_factor = 1.0
            
        return self.regime
    
    def get_dynamic_spacing(self, base_spacing: float, atr: float, price: float) -> float:
        adjusted = base_spacing * self.clustering_factor
        return max(self.config.min_grid_spacing, 
                  min(self.config.max_grid_spacing, adjusted))


class MultiTimeframeMomentum:
    """多时间框架动量"""
    
    def __init__(self, df: pd.DataFrame, config: StrategyConfig):
        self.df = df
        self.config = config
        self.timeframes = config.mtf_periods
        self._precompute()
    
    def _precompute(self):
        for name, period in self.timeframes.items():
            self.df[f'MA_{name}'] = self.df['Close'].rolling(period).mean()
            self.df[f'Momentum_{name}'] = (
                self.df['Close'] / self.df[f'MA_{name}'] - 1
            )
    
    def get_composite_signal(self, idx: int) -> Dict:
        weights = {'short': 0.5, 'medium': 0.3, 'long': 0.2}
        composite = 0
        alignment = 0
        
        for name in self.timeframes.keys():
            mom = self.df[f'Momentum_{name}'].iloc[idx]
            composite += mom * weights[name]
            alignment += 1 if mom > 0 else -1
            
        alignment_score = alignment / len(self.timeframes)
        
        return {
            'composite': composite,
            'alignment': alignment_score,
            'strength': abs(composite),
        }
    
    def should_trade(self, idx: int, direction: str = 'long') -> Tuple[bool, float, float]:
        data = self.get_composite_signal(idx)
        
        if abs(data['alignment']) < 0.0:
            return False, 0.0, data['alignment']
            
        if direction == 'long' and data['composite'] > -0.2:
            return True, data['strength'], data['alignment']
            
        return False, 0.0, data['alignment']


class GridManager:
    """网格管理器"""
    
    def __init__(self, center_price: float, levels: int, spacing: float):
        self.center_price = center_price
        self.levels = levels
        self.spacing = spacing
        self.grids: List[Dict] = []
        self._init_grids()
    
    def _init_grids(self):
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
            })
    
    def update_center(self, new_center: float, spacing: float, 
                     preserve_active: bool = False) -> List[Dict]:
        closed_grids = []
        
        if preserve_active:
            old_grids = {g['level']: g for g in self.grids if g['active']}
            old_center = self.center_price
            price_ratio = new_center / old_center if old_center > 0 else 1.0
            
            self.center_price = new_center
            self.spacing = spacing
            self._init_grids()
            
            for level, old_grid in old_grids.items():
                if level > self.levels:
                    closed_grids.append(old_grid)
                    continue
                
                new_grid = self.grids[level - 1]
                adjusted_entry = old_grid['entry_price'] * price_ratio
                min_profit_price = adjusted_entry * 1.0005
                
                if new_grid['sell_price'] <= min_profit_price:
                    closed_grids.append(old_grid)
                    continue
                
                new_grid['active'] = True
                new_grid['shares'] = old_grid['shares']
                new_grid['entry_price'] = adjusted_entry
                new_grid['entry_date'] = old_grid.get('entry_date')
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


class TrendEnhancedGridStrategyV5_Fixed:
    """修复版v5.0策略"""
    
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
        
        self._restart_phase = 0
        self._restart_base_cash = 0.0
        self._last_restart_idx = 0
        
        self.kelly_sizer = KellyPositionSizer(self.config)
        self.vol_cluster = VolatilityClusteringGrid(self.config)
        self.mtf_momentum: Optional[MultiTimeframeMomentum] = None
        
        self._df: Optional[pd.DataFrame] = None
        self._current_idx: int = 0
        
        self.buy_attempts = 0
        self.buy_executed = 0
        
    def prepare_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values('datetime').reset_index(drop=True)
        
        if df['Volume'].dtype == object:
            def parse_volume(v):
                if isinstance(v, str):
                    v = v.strip().upper()
                    if 'B' in v: return float(v.replace('B', '')) * 1e9
                    if 'M' in v: return float(v.replace('M', '')) * 1e6
                    if 'K' in v: return float(v.replace('K', '')) * 1e3
                return float(v)
            df['Volume'] = df['Volume'].apply(parse_volume)
        
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        tr1 = df['High'] - df['Low']
        tr2 = (df['High'] - df['Close'].shift()).abs()
        tr3 = (df['Low'] - df['Close'].shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(self.config.vol_period).mean()
        
        df['MA5'] = df['Close'].rolling(5).mean()
        df['MA10'] = df['Close'].rolling(10).mean()
        df['MA20'] = df['Close'].rolling(self.config.trend_period).mean()
        
        df['Trend'] = np.where(
            df['Close'] > df['MA5'], 
            (df['Close'] / df['MA20'] - 1).clip(0, 1),
            (df['Close'] / df['MA20'] - 1).clip(-1, 0)
        )
        
        df['Volume_MA20'] = df['Volume'].rolling(20).mean()
        
        # 添加MACD指标
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # 添加布林带
        df['BB_Middle'] = df['Close'].rolling(20).mean()
        bb_std = df['Close'].rolling(20).std()
        df['BB_Upper'] = df['BB_Middle'] + 2 * bb_std
        df['BB_Lower'] = df['BB_Middle'] - 2 * bb_std
        
        # 添加收益率用于计算滚动夏普
        df['returns'] = df['Close'].pct_change()
        
        return df
    
    def get_grid_spacing(self, atr: float, price: float) -> float:
        self.vol_cluster.detect_regime(atr, price, self._current_idx)
        return self.vol_cluster.get_dynamic_spacing(
            self.config.base_grid_pct, atr, price
        )
    
    def apply_costs(self, price: float, is_buy: bool) -> float:
        cost_mult = 1 + self.config.commission_rate + self.config.slippage
        if is_buy:
            return price * cost_mult
        else:
            return price / cost_mult
    
    def execute_buy(self, grid: Dict, price: float, date: pd.Timestamp,
                   total_value: float, is_trend_add: bool = False) -> bool:
        self.buy_attempts += 1
        
        can_trade, strength, alignment = self.mtf_momentum.should_trade(
            self._current_idx, 'long'
        ) if self.mtf_momentum else (True, 0.5, 0)
        
        vol_regime = self.vol_cluster.regime
        trend_score = self._df['Trend'].iloc[self._current_idx]
        
        shares, kelly_pct = self.kelly_sizer.get_position_size(
            total_value, price, vol_regime, trend_score, alignment
        )
        
        if shares <= 0:
            return False
        
        current_pct = self.position * price / total_value
        if current_pct + kelly_pct > self.config.max_position:
            available_pct = self.config.max_position - current_pct
            scale = available_pct / kelly_pct if kelly_pct > 0 else 0
            shares *= scale
        
        executed_price = self.apply_costs(price, True)
        cost = shares * executed_price
        
        cash_reserve = total_value * self.config.min_cash_reserve
        if cost > self.cash - cash_reserve:
            max_cost = self.cash - cash_reserve
            shares = max_cost / executed_price if executed_price > 0 else 0
            cost = shares * executed_price
        
        if shares <= 0 or cost <= 0:
            return False
        
        self.cash -= cost
        self.position += shares
        
        if self.position > 0:
            total_cost = self.cost_basis * (self.position - shares) + cost
            self.cost_basis = total_cost / self.position
        else:
            self.cost_basis = executed_price
        
        self._entry_max_price = max(self._entry_max_price, executed_price)
        
        grid['active'] = True
        grid['shares'] = shares
        grid['entry_price'] = executed_price
        grid['entry_date'] = date
        
        trade_type = 'TREND_ADD' if is_trend_add else 'BUY'
        self.trades.append(Trade(
            date=date, type=trade_type, price=executed_price,
            shares=shares, value=cost, grid_level=grid['level']
        ))
        
        self.buy_executed += 1
        return True
    
    def execute_sell(self, grid: Dict, price: float, date: pd.Timestamp) -> Optional[Trade]:
        if not grid['active'] or grid['shares'] <= 0:
            return None
        
        shares = grid['shares']
        entry_price = grid['entry_price'] if grid['entry_price'] > 0 else grid['buy_price']
        
        executed_price = self.apply_costs(price, False)
        value = shares * executed_price
        
        self.cash += value
        pnl = value - shares * entry_price
        
        self.position -= shares
        if self.position <= 1e-10:
            self.position = 0
            self.cost_basis = 0
            self._entry_max_price = 0
        
        trade = Trade(
            date=date, type='SELL', price=executed_price,
            shares=shares, value=value, pnl=pnl, grid_level=grid['level']
        )
        self.trades.append(trade)
        
        holding_days = (date - grid['entry_date']).days if grid['entry_date'] else 1
        trend_score = self._df['Trend'].iloc[self._current_idx]
        self.kelly_sizer.update_trade(pnl, entry_price, executed_price, 
                                      holding_days, trend_score, self._current_idx)
        
        grid['active'] = False
        grid['shares'] = 0
        grid['entry_price'] = 0
        grid['entry_date'] = None
        
        return trade
    
    def execute_stop_loss(self, price: float, date: pd.Timestamp, 
                         stop_type: str = 'STOP_LOSS') -> Trade:
        if self.position <= 0:
            raise ValueError("无持仓")
        
        executed_price = self.apply_costs(price, False)
        value = self.position * executed_price
        
        self.cash += value
        
        trade = Trade(
            date=date, type=stop_type, price=executed_price,
            shares=self.position, value=value
        )
        self.trades.append(trade)
        
        self.position = 0
        self.cost_basis = 0
        self._entry_max_price = 0
        
        self.cooldown_counter = self.config.cooldown_days
        self._restart_phase = 3
        self._restart_base_cash = self.cash * 0.20
        self._last_restart_idx = self._current_idx
        
        self.kelly_sizer.current_kelly *= 0.7
        
        if self.grid_manager:
            for grid in self.grid_manager.grids:
                if grid['active']:
                    grid['active'] = False
                    grid['shares'] = 0
                    
        return trade
    
    def check_stop_loss(self, total_value: float, price: float, date: pd.Timestamp) -> bool:
        self.max_value = max(self.max_value, total_value)
        
        if self.max_value <= 0:
            return False
        
        drawdown = (self.max_value - total_value) / self.max_value
        
        if drawdown > self.config.stop_loss and self.position > 0:
            self.execute_stop_loss(price, date, 'STOP_LOSS')
            return True
        
        return False
    
    def check_atr_stop(self, price: float, date: pd.Timestamp) -> bool:
        if self.position <= 0 or self._entry_max_price <= 0:
            return False
        
        atr = self._df['ATR'].iloc[self._current_idx]
        stop_price = self._entry_max_price - self.config.atr_stop_multiplier * atr
        
        if price < stop_price:
            self.execute_stop_loss(price, date, 'ATR_STOP')
            return True
        
        return False
    
    def run_backtest(self, df: pd.DataFrame) -> pd.DataFrame:
        self.reset()
        
        print("预计算指标...")
        self._df = self.prepare_indicators(df)
        self.mtf_momentum = MultiTimeframeMomentum(self._df, self.config)
        
        start_idx = max(self.config.trend_period, self.config.vol_period, 60)
        n_bars = len(self._df)
        
        if n_bars <= start_idx:
            raise ValueError(f"数据不足，需要至少{start_idx}条")
        
        init_price = self._df['Close'].iloc[start_idx]
        init_atr = self._df['ATR'].iloc[start_idx]
        spacing = self.get_grid_spacing(init_atr, init_price)
        self.grid_manager = GridManager(init_price, self.config.grid_levels, spacing)
        self.last_grid_update_idx = start_idx
        self._entry_max_price = init_price
        
        print(f"回测区间: {self._df['datetime'].iloc[start_idx]} 至 {self._df['datetime'].iloc[-1]}")
        print(f"初始价格: {init_price:.2f}, 初始网格间距: {spacing:.2%}")
        print(f"初始Kelly: {self.kelly_sizer.current_kelly:.2%}")
        
        for i in range(start_idx, n_bars):
            self._current_idx = i
            
            date = self._df['datetime'].iloc[i]
            price = self._df['Close'].iloc[i]
            high = self._df['High'].iloc[i]
            low = self._df['Low'].iloc[i]
            
            if self.cooldown_counter > 0:
                self.cooldown_counter -= 1
                total = self.cash + self.position * price
                self.max_value = max(self.max_value, total)
                
                self.values.append({
                    'date': date, 'price': price, 'cash': self.cash,
                    'position': self.position, 'value': total,
                    'trend': self._df['Trend'].iloc[i],
                })
                
                if self.cooldown_counter <= self.config.cooldown_days - 2:
                    self._execute_restart(price, date, total)
                continue
            
            self.vol_cluster.detect_regime(self._df['ATR'].iloc[i], price, i)
            
            if (i - self.last_grid_update_idx >= 5):
                if self.should_update_grids(price):
                    self._update_grids(price, date, i)
            
            position_value = self.position * price
            total_value = self.cash + position_value
            
            if self.check_stop_loss(total_value, price, date):
                self._record_value(date, price, total_value, i)
                continue
            
            if self.check_atr_stop(price, date):
                self._record_value(date, price, total_value, i)
                continue
            
            if self.grid_manager:
                buy_signals, sell_signals = self.grid_manager.check_signals(high, low)
                
                for grid in sell_signals:
                    self.execute_sell(grid, price, date)
                
                for grid in buy_signals:
                    self.execute_buy(grid, price, date, total_value)
            
            self._check_trend_add(i, price, date, total_value)
            
            total_value = self.cash + self.position * price
            self._record_value(date, price, total_value, i)
        
        results = pd.DataFrame(self.values)
        
        print(f"\n=== 调试信息 ===")
        print(f"买入尝试: {self.buy_attempts}")
        print(f"买入成功: {self.buy_executed}")
        print(f"总交易数: {len(self.trades)}")
        print(f"最终持仓: {self.position:.2f}")
        print(f"最终现金: {self.cash:.2f}")
        print(f"================\n")
        
        return results
    
    def should_update_grids(self, current_price: float) -> bool:
        if self.grid_manager is None:
            return True
        last_center = self.grid_manager.center_price
        deviation = abs(current_price - last_center) / last_center if last_center > 0 else 0
        return deviation > self.config.position_update_threshold
    
    def _update_grids(self, price: float, date: pd.Timestamp, i: int):
        current_trend = self._df['Trend'].iloc[i]
        old_center = self.grid_manager.center_price
        
        if current_trend > 0:
            new_center = max(old_center, price * 0.97)
        else:
            new_center = min(old_center, price * 1.03)
        
        spacing = self.get_grid_spacing(self._df['ATR'].iloc[i], new_center)
        closed = self.grid_manager.update_center(new_center, spacing, preserve_active=True)
        
        for grid in closed:
            if grid['active'] and grid['shares'] > 0:
                sell_price = self.apply_costs(price, False)
                sell_value = grid['shares'] * sell_price
                self.cash += sell_value
                self.position -= grid['shares']
                pnl = sell_value - grid['shares'] * grid['entry_price']
                
                self.trades.append(Trade(
                    date=date, type='GRID_CLOSE', price=sell_price,
                    shares=grid['shares'], value=sell_value, pnl=pnl
                ))
                
                self.kelly_sizer.update_trade(
                    pnl, grid['entry_price'], sell_price, 1,
                    self._df['Trend'].iloc[i], i
                )
        
        self.last_grid_update_idx = i
    
    def _check_trend_add(self, i: int, price: float, date: pd.Timestamp, total_value: float):
        current_trend = self._df['Trend'].iloc[i]
        
        if current_trend < 0.3 or self.position <= 0:
            return
            
        if (i - self._last_add_idx) < 5:
            return
        
        if self._df['RSI'].iloc[i] > 80:
            return
        
        current_pct = self.position * price / total_value
        if current_pct < self.config.max_position * 0.8:
            inactive = self.grid_manager.get_inactive_buy_grids()
            if inactive:
                if self.execute_buy(inactive[0], price, date, total_value, is_trend_add=True):
                    self._last_add_idx = i
    
    def _execute_restart(self, price: float, date: pd.Timestamp, total_value: float):
        if self._restart_phase <= 0:
            return
        
        if (self._current_idx - self._last_restart_idx) < 3:
            return
        
        spacing = self.get_grid_spacing(self._df['ATR'].iloc[self._current_idx], price)
        levels = max(3, self.config.grid_levels - (3 - self._restart_phase))
        wide_spacing = spacing * (1.3 if self._restart_phase == 3 else 1.1)
        
        self.grid_manager = GridManager(price, levels, wide_spacing)
        
        vol_regime = self.vol_cluster.regime
        trend_score = self._df['Trend'].iloc[self._current_idx]
        
        shares, _ = self.kelly_sizer.get_position_size(
            total_value, price, vol_regime, trend_score * 0.8, 0
        )
        
        invest = shares * price if shares > 0 else self._restart_base_cash
        
        if invest > 0 and invest <= self.cash * 0.95:
            executed_price = self.apply_costs(price, True)
            shares = invest / executed_price
            
            self.cash -= invest
            self.position += shares
            self.cost_basis = executed_price
            self._entry_max_price = executed_price
            
            self.trades.append(Trade(
                date=date, type='RESTART_BUY', price=executed_price,
                shares=shares, value=invest
            ))
        
        self._restart_phase -= 1
        self._last_restart_idx = self._current_idx
        self.last_grid_update_idx = self._current_idx
    
    def _record_value(self, date: pd.Timestamp, price: float, total_value: float, i: int):
        self.values.append({
            'date': date, 'price': price, 'cash': self.cash,
            'position': self.position, 'value': total_value,
            'trend': self._df['Trend'].iloc[i],
            'kelly': self.kelly_sizer.current_kelly,
        })
    
    def get_performance_metrics(self, results: pd.DataFrame) -> Dict:
        if len(results) == 0:
            return {}
        
        initial = self.config.initial_capital
        final = results['value'].iloc[-1]
        total_return = (final - initial) / initial if initial > 0 else 0
        
        days = (results['date'].iloc[-1] - results['date'].iloc[0]).days
        years = days / 365.25 if days > 0 else 0
        annual_return = (1 + total_return) ** (1/years) - 1 if years > 0 else 0
        
        cummax = results['value'].cummax()
        drawdowns = (cummax - results['value']) / cummax
        max_dd = drawdowns.max() if len(cummax) > 0 else 0
        
        daily_ret = results['value'].pct_change().dropna()
        sharpe = ((daily_ret.mean() * 252 - 0.02) / (daily_ret.std() * np.sqrt(252))) \
                 if len(daily_ret) > 1 and daily_ret.std() > 0 else 0
        
        trades_df = pd.DataFrame([{
            'type': t.type, 'pnl': t.pnl, 'value': t.value
        } for t in self.trades])
        
        metrics = {
            'initial_capital': initial,
            'final_value': final,
            'total_return_pct': total_return * 100,
            'annual_return_pct': annual_return * 100,
            'max_drawdown_pct': max_dd * 100,
            'sharpe_ratio': sharpe,
            'calmar_ratio': annual_return / max_dd if max_dd > 0 else 0,
            'total_trades': len(self.trades),
            'final_position': self.position,
            'final_cash': self.cash,
        }
        
        if len(trades_df) > 0:
            sell_trades = trades_df[trades_df['type'].isin(['SELL', 'GRID_CLOSE'])]
            if len(sell_trades) > 0:
                pnls = sell_trades['pnl'].dropna()
                metrics['win_rate_pct'] = (pnls > 0).sum() / len(pnls) * 100 if len(pnls) > 0 else 0
                metrics['avg_pnl'] = pnls.mean()
                metrics['total_pnl'] = pnls.sum()
        
        return metrics
    
    # ========== 升级版可视化方法 ==========
    def plot_results(self, results: pd.DataFrame, save_path: Optional[str] = None, 
                     show_plot: bool = True) -> None:
        """
        绘制专业级回测结果图表（参考优化版v5.0风格）
        """
        if len(results) == 0:
            print("没有数据可供绘图")
            return
            
        # 准备数据
        results = results.copy()
        results['date'] = pd.to_datetime(results['date'])
        results = results.set_index('date')
        
        # 计算指标
        cummax = results['value'].cummax()
        results['drawdown'] = (results['value'] - cummax) / cummax * 100
        results['position_ratio'] = (results['position'] * results['price']) / results['value'] * 100
        
        # 计算Buy&Hold基准
        initial_shares = self.config.initial_capital / results['price'].iloc[0]
        results['buy_hold'] = initial_shares * results['price']
        
        # 计算滚动夏普 (60日)
        results['rolling_sharpe'] = results['value'].pct_change().rolling(60).apply(
            lambda x: (x.mean() * 252 - 0.02) / (x.std() * np.sqrt(252)) if x.std() > 0 else 0
        )
        
        # 计算月收益率热力图数据
        monthly_returns = results['value'].resample('M').last().pct_change() * 100
        monthly_pivot = monthly_returns.groupby([monthly_returns.index.year, monthly_returns.index.month]).last().unstack()
        
        # 创建图表 - 7行2列布局
        fig = plt.figure(figsize=(18, 24))
        gs = GridSpec(7, 2, figure=fig, height_ratios=[2, 1.2, 1, 1, 1, 1, 1], 
                      hspace=0.35, wspace=0.25)
        
        # 颜色定义
        colors = {
            'strategy': '#1f77b4',
            'buyhold': '#ff7f0e',
            'price': '#2ca02c',
            'drawdown': '#d62728',
            'position': '#9467bd',
            'cash': '#8c564b',
            'kelly': '#e377c2',
            'macd': '#7f7f7f',
            'atr': '#bcbd22'
        }
        
        metrics = self.get_performance_metrics(results.reset_index())
        total_ret = metrics.get('total_return_pct', 0)
        max_dd = metrics.get('max_drawdown_pct', 0)
        
        # 1. 主图：策略vs买入持有
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(results.index, results['value'], color=colors['strategy'], 
                linewidth=2, label=f'Optimized Grid Strategy ({total_ret:.1f}%)')
        ax1.plot(results.index, results['buy_hold'], color=colors['buyhold'], 
                linewidth=2, linestyle='--', alpha=0.8, label='Buy & Hold')
        
        # 添加目标线
        target_value = self.config.initial_capital * 1.45
        ax1.axhline(y=target_value, color='green', linestyle='-.', alpha=0.5, label='Target 45%')
        
        # 添加最终收益标注
        final_val = results['value'].iloc[-1]
        buyhold_final = results['buy_hold'].iloc[-1]
        ax1.annotate(f'Strategy: {final_val:,.0f} ({total_ret:.1f}%)', 
                    xy=(results.index[-1], final_val), 
                    xytext=(10, 0), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=colors['strategy'], alpha=0.7),
                    fontsize=9, color='white', fontweight='bold')
        ax1.annotate(f'Buy&Hold: {buyhold_final:,.0f}', 
                    xy=(results.index[-1], buyhold_final), 
                    xytext=(10, 0), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=colors['buyhold'], alpha=0.7),
                    fontsize=9, color='white', fontweight='bold')
        
        ax1.set_title(f'Optimized Trend-Enhanced Smart Grid Strategy v5.0\n'
                     f'Strategy: {total_ret:.1f}% vs Buy&Hold: {(buyhold_final/self.config.initial_capital-1)*100:.1f}% '
                     f'(Target: 45%+)\n'
                     f'Config: Grid {self.config.base_grid_pct*100:.1f}%, Stop {self.config.stop_loss*100:.1f}%, '
                     f'Levels {self.config.grid_levels}', 
                     fontsize=12, fontweight='bold', loc='left')
        ax1.set_ylabel('Portfolio Value (CNY)', fontsize=10)
        ax1.legend(loc='upper left', fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # 2. 股票价格与交易信号
        ax2 = fig.add_subplot(gs[1, :])
        ax2.plot(results.index, results['price'], color='black', linewidth=1.5, label='Stock Price')
        
        # 标记不同类型的交易
        trade_markers = {
            'BUY': ('^', '#2ca02c', 'BUY'),
            'TREND_ADD': ('*', '#1f77b4', 'TREND_ADD'),
            'SELL': ('v', '#d62728', 'SELL'),
            'STOP_LOSS': ('x', '#9467bd', 'STOP_LOSS'),
            'ATR_STOP': ('X', '#8c564b', 'ATR_STOP'),
            'GRID_CLOSE': ('o', '#ff7f0e', 'GRID_CLOSE'),
            'RESTART_BUY': ('s', '#17becf', 'RESTART_BUY')
        }
        
        for trade_type, (marker, color, label) in trade_markers.items():
            trades_of_type = [t for t in self.trades if t.type == trade_type]
            if trades_of_type:
                dates = [t.date for t in trades_of_type]
                prices = [t.price for t in trades_of_type]
                ax2.scatter(dates, prices, marker=marker, c=color, s=60, 
                           label=f'{label} ({len(trades_of_type)})', alpha=0.8, edgecolors='white', linewidth=0.5)
        
        ax2.set_title('Stock Price & Trading Signals (New: Trailing Stop/Profit)', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Price (CNY)', fontsize=10)
        ax2.legend(loc='upper left', fontsize=8, ncol=4)
        ax2.grid(True, alpha=0.3)
        
        # 3. 仓位分配
        ax3 = fig.add_subplot(gs[2, 0])
        ax3.fill_between(results.index, results['position_ratio'], 0, 
                        color=colors['position'], alpha=0.6)
        ax3.plot(results.index, results['position_ratio'], color=colors['position'], linewidth=1)
        ax3.axhline(y=self.config.max_position*100, color='red', linestyle='--', alpha=0.5, label=f'Max {self.config.max_position*100:.0f}%')
        ax3.set_title('Position Allocation', fontsize=11, fontweight='bold')
        ax3.set_ylabel('Position Ratio (%)', fontsize=9)
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)
        
        # 4. 现金储备
        ax4 = fig.add_subplot(gs[2, 1])
        ax4.fill_between(results.index, results['cash'], 0, 
                        color=colors['cash'], alpha=0.6, label='Cash')
        reserve_line = results['value'] * self.config.min_cash_reserve
        ax4.plot(results.index, reserve_line, color='red', linestyle='--', alpha=0.5, label='Reserve Line')
        ax4.set_title('Cash Reserve', fontsize=11, fontweight='bold')
        ax4.set_ylabel('Cash (CNY)', fontsize=9)
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3)
        
        # 5. 趋势指标（多因子）
        ax5 = fig.add_subplot(gs[3, 0])
        if 'trend' in results.columns:
            ax5.plot(results.index, results['trend'], color='#9467bd', linewidth=1, label='Daily Trend')
            # 添加周趋势（20日移动平均）
            weekly_trend = results['trend'].rolling(20).mean()
            ax5.plot(results.index, weekly_trend, color='#d62728', linewidth=1, alpha=0.7, label='Weekly Trend')
            ax5.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax5.fill_between(results.index, results['trend'], 0, 
                           where=(results['trend'] > 0), color='green', alpha=0.2)
            ax5.fill_between(results.index, results['trend'], 0, 
                           where=(results['trend'] < 0), color='red', alpha=0.2)
        ax5.set_title('Trend Indicator (Multi-Factor)', fontsize=11, fontweight='bold')
        ax5.set_ylabel('Trend Strength', fontsize=9)
        ax5.legend(fontsize=8)
        ax5.grid(True, alpha=0.3)
        
        # 6. MACD指标
        ax6 = fig.add_subplot(gs[3, 1])
        if 'MACD' in results.columns and 'MACD_Signal' in results.columns:
            ax6.plot(results.index, results['MACD'], color='blue', linewidth=1, label='MACD')
            ax6.plot(results.index, results['MACD_Signal'], color='red', linewidth=1, label='Signal')
            ax6.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax6.bar(results.index, results['MACD'] - results['MACD_Signal'], 
                   color=['green' if x > 0 else 'red' for x in results['MACD'] - results['MACD_Signal']], 
                   alpha=0.3, width=1)
        ax6.set_title('MACD Indicator', fontsize=11, fontweight='bold')
        ax6.set_ylabel('MACD', fontsize=9)
        ax6.legend(fontsize=8)
        ax6.grid(True, alpha=0.3)
        
        # 7. 回撤分析
        ax7 = fig.add_subplot(gs[4, 0])
        ax7.fill_between(results.index, results['drawdown'], 0, 
                        color=colors['drawdown'], alpha=0.5)
        ax7.plot(results.index, results['drawdown'], color='darkred', linewidth=1)
        # 标记止损点
        stop_loss_trades = [t for t in self.trades if 'STOP' in t.type]
        for t in stop_loss_trades:
            dd_at_date = results.loc[results.index <= t.date, 'drawdown'].iloc[-1] if len(results.loc[results.index <= t.date]) > 0 else 0
            ax7.scatter(t.date, dd_at_date, marker='x', c='darkred', s=50, zorder=5)
        ax7.axhline(y=-max_dd, color='orange', linestyle='--', alpha=0.7)
        ax7.text(results.index[-1], -max_dd, f'Max DD: {max_dd:.1f}%', 
                fontsize=8, va='center', ha='right', 
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        ax7.set_title('Drawdown Analysis (X=Stop Loss)', fontsize=11, fontweight='bold')
        ax7.set_ylabel('Drawdown (%)', fontsize=9)
        ax7.grid(True, alpha=0.3)
        
        # 8. 布林带
        ax8 = fig.add_subplot(gs[4, 1])
        if 'BB_Upper' in results.columns:
            ax8.plot(results.index, results['price'], color='black', linewidth=1, label='Price')
            ax8.plot(results.index, results['BB_Upper'], color='red', linestyle='--', alpha=0.5, label='Upper')
            ax8.plot(results.index, results['BB_Lower'], color='green', linestyle='--', alpha=0.5, label='Lower')
            ax8.fill_between(results.index, results['BB_Upper'], results['BB_Lower'], alpha=0.1, color='gray')
        ax8.set_title('Bollinger Bands', fontsize=11, fontweight='bold')
        ax8.set_ylabel('Price (CNY)', fontsize=9)
        ax8.legend(fontsize=8)
        ax8.grid(True, alpha=0.3)
        
        # 9. PnL分布
        ax9 = fig.add_subplot(gs[5, 0])
        sell_trades = [t for t in self.trades if t.pnl is not None]
        if len(sell_trades) > 0:
            pnls = [t.pnl for t in sell_trades]
            colors_pnl = ['green' if p > 0 else 'red' for p in pnls]
            ax9.bar(range(len(pnls)), pnls, color=colors_pnl, alpha=0.7)
            ax9.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            win_rate = metrics.get('win_rate_pct', 0)
            ax9.set_title(f'PnL Distribution (Win Rate: {win_rate:.1f}%)', fontsize=11, fontweight='bold')
            ax9.set_xlabel('Trade Number', fontsize=9)
            ax9.set_ylabel('PnL (CNY)', fontsize=9)
        ax9.grid(True, alpha=0.3, axis='y')
        
        # 10. 交易统计
        ax10 = fig.add_subplot(gs[5, 1])
        if len(self.trades) > 0:
            trade_type_counts = {}
            for t in self.trades:
                trade_type_counts[t.type] = trade_type_counts.get(t.type, 0) + 1
            
            colors_bar = ['#2ca02c', '#d62728', '#ff7f0e', '#9467bd', '#8c564b', '#17becf', '#e377c2']
            bars = ax10.bar(trade_type_counts.keys(), trade_type_counts.values(), 
                           color=colors_bar[:len(trade_type_counts)], alpha=0.8)
            ax10.set_title('Trade Statistics', fontsize=11, fontweight='bold')
            ax10.set_ylabel('Count', fontsize=9)
            # 在柱子上添加数值
            for bar in bars:
                height = bar.get_height()
                ax10.text(bar.get_x() + bar.get_width()/2., height,
                         f'{int(height)}', ha='center', va='bottom', fontsize=8)
        ax10.grid(True, alpha=0.3, axis='y')
        
        # 11. 滚动夏普比率
        ax11 = fig.add_subplot(gs[6, 0])
        ax11.plot(results.index, results['rolling_sharpe'], color='blue', linewidth=1)
        ax11.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax11.axhline(y=1, color='green', linestyle='--', alpha=0.5)
        ax11.axhline(y=-1, color='red', linestyle='--', alpha=0.5)
        ax11.fill_between(results.index, results['rolling_sharpe'], 0, 
                         where=(results['rolling_sharpe'] > 0), color='green', alpha=0.2)
        ax11.fill_between(results.index, results['rolling_sharpe'], 0, 
                         where=(results['rolling_sharpe'] < 0), color='red', alpha=0.2)
        ax11.set_title('Rolling Sharpe Ratio (60-day)', fontsize=11, fontweight='bold')
        ax11.set_ylabel('Sharpe Ratio', fontsize=9)
        ax11.grid(True, alpha=0.3)
        
        # 12. 月收益率热力图
        ax12 = fig.add_subplot(gs[6, 1])
        if not monthly_pivot.empty:
            sns.heatmap(monthly_pivot, annot=True, fmt='.1f', cmap='RdYlGn', 
                       center=0, ax=ax12, cbar_kws={'label': '%'})
            ax12.set_title('Monthly Returns Heatmap (%)', fontsize=11, fontweight='bold')
            ax12.set_xlabel('Month', fontsize=9)
            ax12.set_ylabel('Year', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
            print(f"图表已保存至: {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()


def print_metrics(metrics: Dict):
    print("\n" + "="*60)
    print("策略绩效报告 - 修复版")
    print("="*60)
    print(f"初始资金: {metrics['initial_capital']:,.0f}")
    print(f"最终价值: {metrics['final_value']:,.0f}")
    print(f"总收益率: {metrics['total_return_pct']:.2f}%")
    print(f"年化收益: {metrics['annual_return_pct']:.2f}%")
    print(f"最大回撤: {metrics['max_drawdown_pct']:.2f}%")
    print(f"夏普比率: {metrics['sharpe_ratio']:.2f}")
    print(f"总交易数: {metrics['total_trades']}")
    print(f"胜率: {metrics.get('win_rate_pct', 0):.1f}%")
    print(f"最终持仓: {metrics['final_position']:.2f}")
    print("="*60)


def main():
    DATA_PATH = 'C:/Users/1/Desktop/python量化/300568历史数据.csv'
    OUTPUT_PATH = 'C:/Users/1/Desktop'
    
    print("趋势增强型网格策略 v5.1 - 修复版")
    
    try:
        df = pd.read_csv(DATA_PATH)
        print(f"数据加载成功: {len(df)} 行")
    except Exception as e:
        print(f"数据加载失败: {e}")
        return
    
    config = StrategyConfig(
        initial_capital=100000,
        base_grid_pct=0.025,
        grid_levels=10,
        max_position=0.95,
        stop_loss=0.20,
        cooldown_days=5,
        min_cash_reserve=0.02,
        kelly_initial=0.15,
        kelly_fraction=0.5,
        commission_rate=0.0003,
        slippage=0.0001
    )
    
    strategy = TrendEnhancedGridStrategyV5_Fixed(config)
    
    try:
        results = strategy.run_backtest(df)
        metrics = strategy.get_performance_metrics(results)
        print_metrics(metrics)
        
        results.to_csv(f"{OUTPUT_PATH}/backtest_results_fixed.csv", index=False)
        print(f"结果已保存至: {OUTPUT_PATH}")
        
        # 生成专业级图表
        chart_path = f"{OUTPUT_PATH}/backtest_chart_v5.png"
        strategy.plot_results(results, save_path=chart_path, show_plot=True)
        
    except Exception as e:
        print(f"回测失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()