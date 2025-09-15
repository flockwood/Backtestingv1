"""Test script to verify transaction cost calculations."""

from core.costs import SimpleCostModel
from core.types import OrderSide

def test_cost_calculations():
    """Test the cost model with example trades."""

    print("="*60)
    print("TESTING TRANSACTION COST CALCULATIONS")
    print("="*60)

    # Create cost model with default parameters
    cost_model = SimpleCostModel(
        fixed_fee=0,
        percentage_fee=0.001,  # 0.1%
        base_slippage_bps=5,
        market_impact_coefficient=10,
        bid_ask_spread_bps=10
    )

    # Test cases
    test_cases = [
        {
            "description": "Small trade with low volume",
            "side": OrderSide.BUY,
            "quantity": 100,
            "price": 150.00,
            "volume": 1_000_000
        },
        {
            "description": "Large trade with market impact",
            "side": OrderSide.BUY,
            "quantity": 10_000,
            "price": 150.00,
            "volume": 1_000_000
        },
        {
            "description": "Trade without volume data",
            "side": OrderSide.SELL,
            "quantity": 500,
            "price": 200.00,
            "volume": None
        }
    ]

    for i, test in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test['description']}")
        print("-" * 40)

        trade_value = test['quantity'] * test['price']
        print(f"Trade Value: ${trade_value:,.2f}")
        print(f"Quantity: {test['quantity']:,} shares")
        print(f"Price: ${test['price']:.2f}")
        print(f"Volume: {test['volume']:,}" if test['volume'] else "Volume: N/A")

        # Calculate costs
        costs = cost_model.calculate_costs(
            side=test['side'],
            quantity=test['quantity'],
            price=test['price'],
            volume=test['volume']
        )

        print("\nCost Breakdown:")
        print(f"  Commission:    ${costs.commission:>10,.2f} ({costs.commission/trade_value*100:.3f}%)")
        print(f"  Slippage:      ${costs.slippage:>10,.2f} ({costs.slippage/trade_value*100:.3f}%)")
        print(f"  Spread Cost:   ${costs.spread_cost:>10,.2f} ({costs.spread_cost/trade_value*100:.3f}%)")
        print(f"  " + "-"*35)
        print(f"  Total Cost:    ${costs.total_cost:>10,.2f} ({costs.total_cost/trade_value*100:.3f}%)")

        # Calculate effective price
        if test['side'] == OrderSide.BUY:
            effective_price = test['price'] * (1 + costs.total_cost/trade_value)
            print(f"\nEffective Buy Price: ${effective_price:.2f} (vs ${test['price']:.2f})")
        else:
            effective_price = test['price'] * (1 - costs.total_cost/trade_value)
            print(f"\nEffective Sell Price: ${effective_price:.2f} (vs ${test['price']:.2f})")

    print("\n" + "="*60)
    print("Cost model testing complete!")
    print("="*60)

if __name__ == "__main__":
    test_cost_calculations()