class TagAdder:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING", {"tooltip": "The prompt"}),
            },
            "optional": {
                "tag1": ("STRING", {"tooltip": "The first"}),
                "tag2": ("STRING", {"tooltip": "The second"}),
                "tag3": ("STRING", {"tooltip": "The third"}),
                "tag4": ("STRING", {"tooltip": "The fourth"}),}

        }

    RETURN_TYPES = ("STRING",)
    OUTPUT_TOOLTIPS = ("The concatenated text output.",)
    FUNCTION = "concatenate_text"
    CATEGORY = "text"

    def concatenate_text(self, prompt, tag1="", tag2="",  tag3="", tag4="", separator=", "):
        # Create a list of tags, stripping whitespace and filtering out empties.
        tags = [tag.strip() for tag in (tag1, tag2, prompt, tag3, tag4) if tag.strip()]
        # Join the list using the given separator.
        result = separator.join(tags)
        return (result,)