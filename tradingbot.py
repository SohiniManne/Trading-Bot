import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
import warnings
import logging
from datetime import datetime, timedelta
import json
import pickle
import os
from typing import Tuple, Dict, Any, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
from concurrent.futures import ThreadPoolExecutor
import smtplib
from email.mime.text import MimeText
import requests
from dataclasses import dataclass
import time

warnings.filterwarnings('ignore')

@dataclass
class RiskMetrics:
    """Risk management metrics"""
    max_position_size: float = 0.25  # Max 25% of portfolio per trade
    stop_loss_pct: float = 0.05      # 5% stop loss
    take_profit_pct: float = 0.10    # 10% take profit
    max_daily_loss: float = 0.03     # Max 3% daily loss
    max_drawdown: float = 0.15       # Max 15% drawdown
    trailing_stop_pct: float = 0.03  # 3% trailing stop

@dataclass
class TradingConfig:
    """Trading configuration"""
    symbol: str
    initial_balance: float = 10000
    commission_rate: float = 0.001   # 0.1% commission
    slippage_rate: float = 0.0005   # 0.05% slippage
    min_trade_amount: float = 100
    max_trades_per_day: int = 5

class AdvancedTechnicalIndicators:
    """Advanced technical indicators"""
    
    @staticmethod
    def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """Williams %R"""
        highest_high = high.rolling(window=window).max()
        lowest_low = low.rolling(window=window).min()
        wr = -100 * (highest_high - close) / (highest_high - lowest_low)
        return wr
    
    @staticmethod
    def cci(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20) -> pd.Series:
        """Commodity Channel Index"""
        tp = (high + low + close) / 3
        ma = tp.rolling(window=window).mean()
        md = tp.rolling(window=window).apply(lambda x: np.mean(np.abs(x - x.mean())))
        cci = (tp - ma) / (0.015 * md)
        return cci
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """Average True Range"""
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        return true_range.rolling(window=window).mean()
    
    @staticmethod
    def adx(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """Average Directional Index"""
        plus_dm = high.diff()
        minus_dm = low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        
        tr = AdvancedTechnicalIndicators.atr(high, low, close, 1)
        plus_di = 100 * (plus_dm.rolling(window=window).mean() / tr.rolling(window=window).mean())
        minus_di = 100 * (minus_dm.abs().rolling(window=window).mean() / tr.rolling(window=window).mean())
        
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=window).mean()
        return adx
    
    @staticmethod
    def vwap(price: pd.Series, volume: pd.Series) -> pd.Series:
        """Volume Weighted Average Price"""
        return (price * volume).cumsum() / volume.cumsum()
    
    @staticmethod
    def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """On Balance Volume"""
        return (np.sign(close.diff()) * volume).fillna(0).cumsum()
    
    @staticmethod
    def mfi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, window: int = 14) -> pd.Series:
        """Money Flow Index"""
        tp = (high + low + close) / 3
        rmf = tp * volume
        
        positive_flow = pd.Series(np.where(tp > tp.shift(1), rmf, 0), index=tp.index)
        negative_flow = pd.Series(np.where(tp < tp.shift(1), rmf, 0), index=tp.index)
        
        positive_mf = positive_flow.rolling(window=window).sum()
        negative_mf = negative_flow.rolling(window=window).sum()
        
        mfi = 100 - (100 / (1 + positive_mf / negative_mf))
        return mfi

class PatternRecognition:
    """Chart pattern recognition"""
    
    @staticmethod
    def detect_support_resistance(data: pd.DataFrame, window: int = 20) -> Dict[str, List[float]]:
        """Detect support and resistance levels"""
        highs = data['High'].rolling(window=window, center=True).max()
        lows = data['Low'].rolling(window=window, center=True).min()
        
        resistance_levels = data[data['High'] == highs]['High'].dropna().unique()
        support_levels = data[data['Low'] == lows]['Low'].dropna().unique()
        
        return {
            'resistance': sorted(resistance_levels[-5:], reverse=True),  # Top 5
            'support': sorted(support_levels[-5:])  # Bottom 5
        }
    
    @staticmethod
    def detect_trend(close: pd.Series, window: int = 20) -> str:
        """Detect trend direction"""
        sma_short = close.rolling(window=window//2).mean()
        sma_long = close.rolling(window=window).mean()
        
        if sma_short.iloc[-1] > sma_long.iloc[-1] and sma_short.iloc[-2] > sma_long.iloc[-2]:
            return "UPTREND"
        elif sma_short.iloc[-1] < sma_long.iloc[-1] and sma_short.iloc[-2] < sma_long.iloc[-2]:
            return "DOWNTREND"
        else:
            return "SIDEWAYS"
    
    @staticmethod
    def detect_candlestick_patterns(data: pd.DataFrame) -> Dict[str, bool]:
        """Detect common candlestick patterns"""
        patterns = {}
        
        # Doji
        body = abs(data['Close'] - data['Open'])
        range_size = data['High'] - data['Low']
        patterns['doji'] = (body / range_size).iloc[-1] < 0.1
        
        # Hammer
        lower_shadow = data['Low'] - np.minimum(data['Open'], data['Close'])
        upper_shadow = data['High'] - np.maximum(data['Open'], data['Close'])
        patterns['hammer'] = (lower_shadow / body).iloc[-1] > 2 and (upper_shadow / body).iloc[-1] < 0.5
        
        # Engulfing patterns
        patterns['bullish_engulfing'] = (
            data['Close'].iloc[-2] < data['Open'].iloc[-2] and  # Previous red candle
            data['Close'].iloc[-1] > data['Open'].iloc[-1] and  # Current green candle
            data['Open'].iloc[-1] < data['Close'].iloc[-2] and  # Current opens below previous close
            data['Close'].iloc[-1] > data['Open'].iloc[-2]     # Current closes above previous open
        )
        
        return patterns

class SentimentAnalysis:
    """Market sentiment analysis"""
    
    @staticmethod
    def get_fear_greed_index() -> Dict[str, Any]:
        """Get Fear & Greed Index (mock implementation)"""
        # In real implementation, you'd call CNN's API or scrape data
        return {
            'value': np.random.randint(0, 100),
            'classification': 'Neutral',
            'timestamp': datetime.now()
        }
    
    @staticmethod
    def analyze_news_sentiment(symbol: str) -> Dict[str, float]:
        """Analyze news sentiment (mock implementation)"""
        # In real implementation, you'd use news APIs and NLP
        return {
            'sentiment_score': np.random.uniform(-1, 1),
            'news_count': np.random.randint(5, 50),
            'positive_ratio': np.random.uniform(0.3, 0.7)
        }

class RiskManager:
    """Advanced risk management"""
    
    def __init__(self, config: RiskMetrics):
        self.config = config
        self.daily_pnl = 0
        self.peak_portfolio_value = 0
        self.trades_today = 0
        self.last_trade_date = None
        
    def calculate_position_size(self, portfolio_value: float, volatility: float, 
                              confidence: float) -> float:
        """Calculate optimal position size using Kelly Criterion"""
        # Simplified Kelly Criterion
        win_prob = 0.5 + (confidence * 0.3)  # Adjust based on confidence
        avg_win = 0.05  # Average 5% win
        avg_loss = 0.03  # Average 3% loss
        
        kelly_fraction = (win_prob * avg_win - (1 - win_prob) * avg_loss) / avg_win
        kelly_fraction = max(0, min(kelly_fraction, self.config.max_position_size))
        
        # Adjust for volatility
        volatility_adj = min(1.0, 0.02 / max(volatility, 0.01))
        
        position_size = kelly_fraction * volatility_adj * portfolio_value
        return min(position_size, portfolio_value * self.config.max_position_size)
    
    def should_stop_trading(self, current_portfolio_value: float) -> Tuple[bool, str]:
        """Check if trading should be stopped"""
        current_date = datetime.now().date()
        
        # Reset daily counters
        if self.last_trade_date != current_date:
            self.trades_today = 0
            self.daily_pnl = 0
            self.last_trade_date = current_date
        
        # Check daily loss limit
        if self.daily_pnl < -self.config.max_daily_loss * current_portfolio_value:
            return True, "Daily loss limit exceeded"
        
        # Check maximum drawdown
        if self.peak_portfolio_value > 0:
            drawdown = (self.peak_portfolio_value - current_portfolio_value) / self.peak_portfolio_value
            if drawdown > self.config.max_drawdown:
                return True, f"Maximum drawdown exceeded: {drawdown:.2%}"
        
        # Check daily trade limit
        if self.trades_today >= 5:  # Max trades per day
            return True, "Daily trade limit exceeded"
        
        return False, ""
    
    def update_portfolio_peak(self, current_value: float):
        """Update peak portfolio value"""
        self.peak_portfolio_value = max(self.peak_portfolio_value, current_value)

class MultiTimeframeAnalysis:
    """Multi-timeframe analysis"""
    
    @staticmethod
    def analyze_multiple_timeframes(symbol: str, timeframes: List[str]) -> Dict[str, Dict]:
        """Analyze multiple timeframes"""
        results = {}
        
        for timeframe in timeframes:
            try:
                data = yf.Ticker(symbol).history(period=timeframe)
                if not data.empty:
                    # Calculate basic indicators for each timeframe
                    data['sma_20'] = data['Close'].rolling(20).mean()
                    data['rsi'] = calculate_rsi(data['Close'])
                    
                    results[timeframe] = {
                        'trend': 'UP' if data['Close'].iloc[-1] > data['sma_20'].iloc[-1] else 'DOWN',
                        'rsi': data['rsi'].iloc[-1],
                        'momentum': (data['Close'].iloc[-1] / data['Close'].iloc[-5] - 1) * 100
                    }
            except Exception as e:
                logging.error(f"Error analyzing {timeframe}: {e}")
                
        return results

class DatabaseManager:
    """Database management for storing trading data"""
    
    def __init__(self, db_path: str = "trading_data.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Trades table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                symbol TEXT,
                action TEXT,
                shares INTEGER,
                price REAL,
                total_cost REAL,
                commission REAL,
                portfolio_value REAL,
                reasoning TEXT
            )
        ''')
        
        # Portfolio history table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS portfolio_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                symbol TEXT,
                portfolio_value REAL,
                balance REAL,
                shares_held INTEGER,
                stock_price REAL,
                daily_return REAL
            )
        ''')
        
        # Model performance table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                symbol TEXT,
                model_type TEXT,
                accuracy REAL,
                precision_score REAL,
                recall REAL,
                sharpe_ratio REAL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_trade(self, trade_data: Dict):
        """Save trade to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO trades (timestamp, symbol, action, shares, price, total_cost, commission, portfolio_value, reasoning)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            trade_data['timestamp'],
            trade_data.get('symbol', ''),
            trade_data['action'],
            trade_data['shares'],
            trade_data['price'],
            trade_data['total'],
            trade_data.get('commission', 0),
            trade_data['portfolio_value'],
            trade_data.get('reasoning', '')
        ))
        
        conn.commit()
        conn.close()

class NotificationManager:
    """Handle notifications via email, SMS, etc."""
    
    def __init__(self, email_config: Dict = None):
        self.email_config = email_config or {}
    
    def send_trade_alert(self, trade_info: Dict):
        """Send trade alert"""
        message = f"""
        TRADE EXECUTED:
        Action: {trade_info['action']}
        Symbol: {trade_info.get('symbol', 'N/A')}
        Shares: {trade_info['shares']}
        Price: ${trade_info['price']:.2f}
        Total: ${trade_info['total']:.2f}
        Reasoning: {trade_info.get('reasoning', 'N/A')}
        """
        
        self._send_email("Trade Alert", message)
    
    def send_risk_alert(self, alert_message: str):
        """Send risk management alert"""
        self._send_email("Risk Alert", alert_message)
    
    def _send_email(self, subject: str, message: str):
        """Send email notification"""
        if not self.email_config:
            print(f"EMAIL ALERT - {subject}: {message}")
            return
        
        try:
            # Email sending logic would go here
            print(f"Email sent - {subject}: {message}")
        except Exception as e:
            logging.error(f"Failed to send email: {e}")

class PortfolioOptimizer:
    """Portfolio optimization using Modern Portfolio Theory"""
    
    @staticmethod
    def calculate_optimal_weights(returns: pd.DataFrame, risk_tolerance: float = 0.5) -> Dict[str, float]:
        """Calculate optimal portfolio weights"""
        # Simplified mean-variance optimization
        mean_returns = returns.mean()
        cov_matrix = returns.cov()
        
        # Risk parity approach (simplified)
        inv_vol = 1 / returns.std()
        weights = inv_vol / inv_vol.sum()
        
        return weights.to_dict()
    
    @staticmethod
    def rebalance_portfolio(current_weights: Dict[str, float], 
                          target_weights: Dict[str, float],
                          threshold: float = 0.05) -> Dict[str, str]:
        """Determine rebalancing actions"""
        actions = {}
        
        for symbol in target_weights:
            current = current_weights.get(symbol, 0)
            target = target_weights[symbol]
            diff = abs(current - target)
            
            if diff > threshold:
                if current < target:
                    actions[symbol] = "BUY"
                else:
                    actions[symbol] = "SELL"
        
        return actions

class AdvancedMLModels:
    """Advanced machine learning models"""
    
    def __init__(self):
        self.models = {
            'gradient_boost': GradientBoostingRegressor(n_estimators=200, learning_rate=0.1),
            'random_forest': RandomForestRegressor(n_estimators=300, max_depth=10),
            'ensemble': None  # Will be created later
        }
        self.scalers = {name: RobustScaler() for name in self.models.keys()}
        
    def create_ensemble(self):
        """Create ensemble model"""
        from sklearn.ensemble import VotingRegressor
        
        estimators = [(name, model) for name, model in self.models.items() if model is not None]
        self.models['ensemble'] = VotingRegressor(estimators=estimators)
    
    def train_with_cross_validation(self, X: np.ndarray, y: np.ndarray, cv_folds: int = 5):
        """Train models with time series cross validation"""
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        results = {}
        
        for name, model in self.models.items():
            if model is None:
                continue
                
            scores = []
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Scale features
                X_train_scaled = self.scalers[name].fit_transform(X_train)
                X_val_scaled = self.scalers[name].transform(X_val)
                
                # Train and evaluate
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_val_scaled)
                score = r2_score(y_val, y_pred)
                scores.append(score)
            
            results[name] = {
                'mean_score': np.mean(scores),
                'std_score': np.std(scores)
            }
        
        return results

class MarketRegimeDetection:
    """Detect market regimes (bull/bear/sideways)"""
    
    @staticmethod
    def detect_regime(returns: pd.Series, window: int = 60) -> pd.Series:
        """Detect market regime using rolling statistics"""
        rolling_mean = returns.rolling(window=window).mean()
        rolling_std = returns.rolling(window=window).std()
        rolling_sharpe = rolling_mean / rolling_std
        
        regimes = pd.Series(index=returns.index, dtype=str)
        
        # Define regime thresholds
        bull_threshold = 0.1
        bear_threshold = -0.1
        
        regimes[rolling_sharpe > bull_threshold] = 'BULL'
        regimes[rolling_sharpe < bear_threshold] = 'BEAR'
        regimes[(rolling_sharpe >= bear_threshold) & (rolling_sharpe <= bull_threshold)] = 'SIDEWAYS'
        
        return regimes.fillna('SIDEWAYS')

class AdvancedTradingBot:
    """Enhanced trading bot with all advanced features"""
    
    def __init__(self, config: TradingConfig, risk_config: RiskMetrics):
        self.config = config
        self.risk_config = risk_config
        
        # Core components
        self.data_processor = None
        self.ml_models = AdvancedMLModels()
        self.risk_manager = RiskManager(risk_config)
        self.db_manager = DatabaseManager()
        self.notification_manager = NotificationManager()
        self.pattern_recognizer = PatternRecognition()
        self.sentiment_analyzer = SentimentAnalysis()
        
        # Trading state
        self.portfolio = {
            'balance': config.initial_balance,
            'positions': {},
            'total_value': config.initial_balance
        }
        
        # Performance tracking
        self.performance_metrics = {}
        self.trades_history = []
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def execute_advanced_strategy(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Execute advanced trading strategy with all features"""
        
        # 1. Multi-timeframe analysis
        mtf_analysis = MultiTimeframeAnalysis.analyze_multiple_timeframes(
            self.config.symbol, ['1mo', '3mo', '6mo']
        )
        
        # 2. Pattern recognition
        patterns = self.pattern_recognizer.detect_candlestick_patterns(market_data)
        support_resistance = self.pattern_recognizer.detect_support_resistance(market_data)
        trend = self.pattern_recognizer.detect_trend(market_data['Close'])
        
        # 3. Sentiment analysis
        sentiment = self.sentiment_analyzer.analyze_news_sentiment(self.config.symbol)
        fear_greed = self.sentiment_analyzer.get_fear_greed_index()
        
        # 4. Market regime detection
        regime = MarketRegimeDetection.detect_regime(market_data['Close'].pct_change())
        current_regime = regime.iloc[-1] if not regime.empty else 'SIDEWAYS'
        
        # 5. Risk management check
        should_stop, stop_reason = self.risk_manager.should_stop_trading(
            self.portfolio['total_value']
        )
        
        if should_stop:
            self.notification_manager.send_risk_alert(stop_reason)
            return {'action': 'HOLD', 'reason': stop_reason}
        
        # 6. Generate trading signal with all factors
        signal_strength = 0
        reasons = []
        
        # Technical factors
        if trend == 'UPTREND':
            signal_strength += 1
            reasons.append("Uptrend detected")
        elif trend == 'DOWNTREND':
            signal_strength -= 1
            reasons.append("Downtrend detected")
        
        # Pattern factors
        if patterns.get('bullish_engulfing', False):
            signal_strength += 1
            reasons.append("Bullish engulfing pattern")
        
        # Sentiment factors
        if sentiment['sentiment_score'] > 0.3:
            signal_strength += 0.5
            reasons.append("Positive news sentiment")
        elif sentiment['sentiment_score'] < -0.3:
            signal_strength -= 0.5
            reasons.append("Negative news sentiment")
        
        # Regime adjustment
        if current_regime == 'BULL':
            signal_strength *= 1.2
        elif current_regime == 'BEAR':
            signal_strength *= 0.8
        
        # Determine action
        if signal_strength >= 2:
            action = 'BUY'
        elif signal_strength <= -2:
            action = 'SELL'
        else:
            action = 'HOLD'
        
        return {
            'action': action,
            'signal_strength': signal_strength,
            'reasons': reasons,
            'mtf_analysis': mtf_analysis,
            'patterns': patterns,
            'sentiment': sentiment,
            'regime': current_regime,
            'support_resistance': support_resistance
        }

# Utility function
def calculate_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    """Calculate RSI"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def main_advanced():
    """Main function for advanced trading bot"""
    # Configuration
    config = TradingConfig(
        symbol="AAPL",
        initial_balance=50000,
        commission_rate=0.001
    )
    
    risk_config = RiskMetrics(
        max_position_size=0.25,
        stop_loss_pct=0.05,
        take_profit_pct=0.10
    )
    
    # Initialize advanced bot
    bot = AdvancedTradingBot(config, risk_config)
    
    try:
        # Load data
        data = yf.Ticker(config.symbol).history(period="1y")
        
        # Execute strategy
        result = bot.execute_advanced_strategy(data)
        
        print("=== ADVANCED TRADING SIGNAL ===")
        print(f"Action: {result['action']}")
        print(f"Signal Strength: {result['signal_strength']}")
        print(f"Reasons: {', '.join(result['reasons'])}")
        print(f"Current Regime: {result['regime']}")
        print(f"Sentiment Score: {result['sentiment']['sentiment_score']:.3f}")
        
    except Exception as e:
        logging.error(f"Error in advanced trading bot: {e}")

if __name__ == "__main__":
    main_advanced()