class AnyType(str):
    """
    ComfyUI wildcard type that accepts connections from any socket type.
    """

    def __ne__(self, other):
        return False


ANY_TYPE = AnyType("*")


class HoneyGetListRange:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "data": (
                    ANY_TYPE,
                    {
                        "tooltip": "List containing any ComfyUI data type.",
                    },
                ),
                "index": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "step": 1,
                        "tooltip": "Zero-based index of the first item.",
                    },
                ),
                "number_of_items": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "step": 1,
                        "tooltip": "Number of consecutive items to return.",
                    },
                ),
            }
        }

    # Receive the complete upstream list rather than letting ComfyUI
    # execute this node separately for every item.
    INPUT_IS_LIST = True

    RETURN_TYPES = (ANY_TYPE, "INT")
    RETURN_NAMES = ("selected_items", "selected_count")

    # selected_items is still a list; selected_count is one integer.
    OUTPUT_IS_LIST = (True, False)

    FUNCTION = "get_range"
    CATEGORY = "Honey/List"

    def get_range(self, data, index, number_of_items):
        start = int(index[0]) if isinstance(index, list) else int(index)
        requested = (
            int(number_of_items[0])
            if isinstance(number_of_items, list)
            else int(number_of_items)
        )

        if not isinstance(data, list):
            data = [data]

        total = len(data)

        if total == 0:
            return ([], 0)

        if start < 0:
            raise ValueError(
                f"Index cannot be negative. Received index {start}."
            )

        if start >= total:
            # Nothing remains after this index.
            return ([], 0)

        # Limit the result to however many items actually remain.
        remaining = total - start
        amount_to_return = min(requested, remaining)
        end = start + amount_to_return

        selected = data[start:end]

        print(
            f"[Honey Get List Range] Requested {requested} item(s) "
            f"starting at index {start}. Returned {len(selected)} item(s)."
        )

        return (selected, len(selected))


NODE_CLASS_MAPPINGS = {
    "HoneyGetListRange": HoneyGetListRange,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HoneyGetListRange": "List Killer",
}