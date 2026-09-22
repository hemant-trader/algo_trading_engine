class PulseTTLStateMachine:
    def __init__(self, max_ttl: int = 5):
        self.max_ttl = max_ttl
        self.current_ttl = 0

    def refresh(self):
        self.current_ttl = self.max_ttl

    def pulse(self, valid_fresh_tick: bool) -> int:
        if valid_fresh_tick:
            self.refresh()
        else:
            if self.current_ttl > 0:
                self.current_ttl -= 1
        return self.current_ttl

    def is_alive(self) -> bool:
        return self.current_ttl > 0