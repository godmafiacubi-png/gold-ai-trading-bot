from __future__ import annotations


class ExecutionEngine:
    def __init__(self, config: dict, logger):
        self.config = config
        self.logger = logger
        self.dry_run = bool(config["execution"].get("dry_run", True))

    def execute(self, order: dict) -> dict:
        if order.get("action") == "NO_TRADE":
            return {"status": "SKIPPED", "order": order}

        execution_plan = order.get("execution_plan") or {}
        if execution_plan:
            self.logger.info("Execution plan: %s", execution_plan)

        if self.dry_run:
            self.logger.info("DRY RUN order: %s", order)
            return {"status": "DRY_RUN", "order": order}

        return {
            "status": "BLOCKED",
            "reason": "Real execution is intentionally not implemented in v1 scaffold.",
            "order": order,
        }
