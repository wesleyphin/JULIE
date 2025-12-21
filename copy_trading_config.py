"""
Copy Trading Configuration Management
======================================

Manages persistent copy trading configuration without requiring config.py edits.
Stores configuration in copy_trading_config.json for runtime enable/disable.

Author: Wes (with Claude)
"""

import json
import os
import logging
from typing import Dict, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

CONFIG_FILE = Path(__file__).parent / "copy_trading_config.json"


def load_copy_trading_config() -> Dict:
    """
    Load copy trading configuration from JSON file.

    Returns:
        Dict with keys: enabled, followers
    """
    if not CONFIG_FILE.exists():
        return {
            'enabled': False,
            'followers': []
        }

    try:
        with open(CONFIG_FILE, 'r') as f:
            config = json.load(f)
            logger.info(f"✅ Loaded copy trading config: {len(config.get('followers', []))} followers, enabled={config.get('enabled', False)}")
            return config
    except Exception as e:
        logger.error(f"Error loading copy trading config: {e}")
        return {
            'enabled': False,
            'followers': []
        }


def save_copy_trading_config(enabled: bool, followers: List[Dict]) -> bool:
    """
    Save copy trading configuration to JSON file.

    Args:
        enabled: Whether copy trading is enabled
        followers: List of follower account dicts

    Returns:
        True if saved successfully
    """
    config = {
        'enabled': enabled,
        'followers': followers
    }

    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info(f"✅ Saved copy trading config: {len(followers)} followers, enabled={enabled}")
        return True
    except Exception as e:
        logger.error(f"Error saving copy trading config: {e}")
        return False


def enable_copy_trading() -> bool:
    """
    Enable copy trading (keeps existing followers).

    Returns:
        True if enabled successfully
    """
    config = load_copy_trading_config()
    config['enabled'] = True
    return save_copy_trading_config(config['enabled'], config['followers'])


def disable_copy_trading() -> bool:
    """
    Disable copy trading (keeps followers configured for later).

    Returns:
        True if disabled successfully
    """
    config = load_copy_trading_config()
    config['enabled'] = False
    return save_copy_trading_config(config['enabled'], config['followers'])


def update_followers(followers: List[Dict]) -> bool:
    """
    Update follower accounts configuration.

    Args:
        followers: List of follower account dicts with keys:
            - account_id
            - username
            - api_key
            - contract_id
            - size_ratio
            - enabled

    Returns:
        True if updated successfully
    """
    config = load_copy_trading_config()
    config['followers'] = followers
    return save_copy_trading_config(config['enabled'], followers)


def get_copy_trading_status() -> Dict:
    """
    Get current copy trading status.

    Returns:
        Dict with keys: enabled, follower_count, followers
    """
    config = load_copy_trading_config()
    active_followers = [f for f in config.get('followers', []) if f.get('enabled', True)]

    return {
        'enabled': config.get('enabled', False),
        'follower_count': len(active_followers),
        'followers': active_followers,
        'all_followers': config.get('followers', [])
    }


def clear_copy_trading_config() -> bool:
    """
    Clear all copy trading configuration.

    Returns:
        True if cleared successfully
    """
    return save_copy_trading_config(False, [])


def has_copy_trading_config() -> bool:
    """
    Check if copy trading configuration exists.

    Returns:
        True if config file exists and has followers
    """
    if not CONFIG_FILE.exists():
        return False

    config = load_copy_trading_config()
    return len(config.get('followers', [])) > 0
