"""Bulletin board system for agents to share trading information."""
import datetime
from typing import Dict, List


class BulletinBoardSystem:
    """Implements a bulletin board system for agents to share trading tips."""
    def __init__(self):
        self.daily_messages = {}  # day -> list of messages

    def post_message(self, day: int, agent_id: int, message: str):
        """Post a message to the bulletin board."""
        if day not in self.daily_messages:
            self.daily_messages[day] = []

        self.daily_messages[day].append({
            "agent_id": agent_id,
            "message": message,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

    def get_messages_for_day(self, day: int) -> List[Dict]:
        """Get all messages posted on a specific day."""
        return self.daily_messages.get(day, [])

    def get_recent_messages(self, current_day: int, num_days: int = 5) -> List[Dict]:
        """Get messages from the last N days."""
        messages = []
        for day in range(max(1, current_day - num_days), current_day):
            messages.extend(self.get_messages_for_day(day))
        return messages