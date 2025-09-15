from abc import ABC, abstractmethod
from typing import Optional
from .types import OrderSide, CostStructure


class CostModel(ABC):
    """Base class for transaction cost models."""

    @abstractmethod
    def calculate_costs(
        self,
        side: OrderSide,
        quantity: float,
        price: float,
        volume: Optional[float] = None
    ) -> CostStructure:
        """
        Calculate transaction costs for a trade.

        Args:
            side: Buy or sell side
            quantity: Number of shares
            price: Price per share
            volume: Average daily volume (optional, for slippage calculation)

        Returns:
            CostStructure with breakdown of costs
        """
        pass


class SimpleCostModel(CostModel):
    """Simple transaction cost model with commission, slippage, and spread."""

    def __init__(
        self,
        fixed_fee: float = 0.0,
        percentage_fee: float = 0.001,  # 10 bps = 0.1%
        base_slippage_bps: float = 5,  # basis points
        market_impact_coefficient: float = 10,  # basis points
        bid_ask_spread_bps: float = 10  # basis points
    ):
        """
        Initialize SimpleCostModel.

        Args:
            fixed_fee: Fixed commission per trade
            percentage_fee: Commission as percentage of trade value (e.g., 0.001 = 0.1%)
            base_slippage_bps: Base slippage in basis points
            market_impact_coefficient: Market impact coefficient in basis points
            bid_ask_spread_bps: Bid-ask spread in basis points
        """
        self.fixed_fee = fixed_fee
        self.percentage_fee = percentage_fee
        self.base_slippage_bps = base_slippage_bps
        self.market_impact_coefficient = market_impact_coefficient
        self.bid_ask_spread_bps = bid_ask_spread_bps

        # Convert basis points to decimal
        self.base_slippage = base_slippage_bps / 10000
        self.market_impact_coeff = market_impact_coefficient / 10000
        self.bid_ask_spread = bid_ask_spread_bps / 10000

    def calculate_costs(
        self,
        side: OrderSide,
        quantity: float,
        price: float,
        volume: Optional[float] = None
    ) -> CostStructure:
        """
        Calculate transaction costs including commission, slippage, and spread.

        Args:
            side: Buy or sell side
            quantity: Number of shares
            price: Price per share
            volume: Average daily volume (optional, for slippage calculation)

        Returns:
            CostStructure with breakdown of costs
        """
        trade_value = quantity * price

        # Calculate commission
        commission = self.fixed_fee + (trade_value * self.percentage_fee)

        # Calculate slippage
        if volume and volume > 0:
            # Market impact increases with order size relative to volume
            market_impact = (quantity / volume) * self.market_impact_coeff
            slippage_rate = self.base_slippage + market_impact
        else:
            # If no volume data, use base slippage only
            slippage_rate = self.base_slippage

        slippage_cost = trade_value * slippage_rate

        # Calculate bid-ask spread cost (half spread for each side)
        spread_cost = trade_value * (self.bid_ask_spread / 2)

        # Total cost
        total_cost = commission + slippage_cost + spread_cost

        return CostStructure(
            commission=commission,
            slippage=slippage_cost,
            spread_cost=spread_cost,
            total_cost=total_cost
        )


class ZeroCostModel(CostModel):
    """Zero cost model for testing without transaction costs."""

    def calculate_costs(
        self,
        side: OrderSide,
        quantity: float,
        price: float,
        volume: Optional[float] = None
    ) -> CostStructure:
        """Return zero costs."""
        return CostStructure(
            commission=0.0,
            slippage=0.0,
            spread_cost=0.0,
            total_cost=0.0
        )