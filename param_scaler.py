import logging

# Configuration
MIN_SL_FLOOR = 4.0

def apply_scaling():
    """
    Scans loaded regime parameters and mathematically scales any SL < 4.0
    up to 4.0, applying the exact same multiplier to the TP.
    """
    try:
        import regime_sltp_params
    except ImportError:
        logging.warning("⚠️ Param Scaler: 'regime_sltp_params.py' not found. Skipping scaling.")
        return

    count = 0
    # Iterate over the dictionary directly (Pass by Reference allows in-place edit)
    for combo_key, directions in regime_sltp_params.PARAMS.items():
        for side, params in directions.items():
            current_sl = params.get('sl', 0)
            current_tp = params.get('tp', 0)

            # If SL is unsafe (< 4.0), scale everything up
            if current_sl < MIN_SL_FLOOR and current_sl > 0:
                # 1. Calculate the Multiplier required to make SL safe
                ratio = MIN_SL_FLOOR / current_sl

                # 2. Calculate new targets
                new_sl = MIN_SL_FLOOR
                new_tp = round(current_tp * ratio, 2)

                # 3. Update the Dictionary In-Place
                params['sl'] = new_sl
                params['tp'] = new_tp

                count += 1

    if count > 0:
        print(f"\n⚖️  PARAM SCALER: Automatically adjusted {count} regime parameters.")
        print(f"    (All stops now ≥ {MIN_SL_FLOOR} pts while maintaining exact R:R ratios)")
    else:
        print("\n✅ PARAM SCALER: All parameters are already compliant.")

if __name__ == "__main__":
    # Allow running this script standalone to test
    logging.basicConfig(level=logging.INFO)
    apply_scaling()
