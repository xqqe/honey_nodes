import re
from collections import OrderedDict


class CombineRepeatedVariables:
    """
    Merges repeated Folded Prompt / Dynamic Prompt variable definitions
    and optionally adds pipe separators inside each variable block.

    Example input:

        ${acc={
        __WC-ACC-TECH-PROP__,
        __WC-ACC-WILDWEST-PROP__,
        }}

        ${outfit={
        __COPS_BP-GOLDEN-ARTISTIC__
        __COPS_BP-GOLDEN-BASE-LINGERIE__
        }}

    Example output:

        ${acc={
        __WC-ACC-TECH-PROP__,|
        __WC-ACC-WILDWEST-PROP__,
        }}

        ${outfit={
        __COPS_BP-GOLDEN-ARTISTIC__|
        __COPS_BP-GOLDEN-BASE-LINGERIE__
        }}
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {
                    "multiline": True,
                    "dynamicPrompts": False,
                    "default": ""
                }),
                "merge_repeated_variables": ("BOOLEAN", {
                    "default": True
                }),
                "add_pipe_separators": ("BOOLEAN", {
                    "default": True
                }),
                "dedupe_lines": ("BOOLEAN", {
                    "default": False
                }),
                "keep_non_variable_text": ("BOOLEAN", {
                    "default": True
                }),
                "blank_line_between_blocks": ("BOOLEAN", {
                    "default": False
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "combine"
    CATEGORY = "Honey"

    def combine(
        self,
        text: str,
        merge_repeated_variables: bool = True,
        add_pipe_separators: bool = True,
        dedupe_lines: bool = False,
        keep_non_variable_text: bool = True,
        blank_line_between_blocks: bool = False,
    ):
        if not merge_repeated_variables:
            result = text
            if add_pipe_separators:
                result = self._add_pipe_separators_to_existing_text(result)
            return (result,)

        merged = OrderedDict()

        blocks, consumed_ranges = self._extract_variable_blocks(text)

        for var_name, body in blocks:
            if var_name not in merged:
                merged[var_name] = []

            body_lines = body.splitlines()

            for line in body_lines:
                clean_line = line.rstrip("\r")

                if dedupe_lines:
                    if clean_line not in merged[var_name]:
                        merged[var_name].append(clean_line)
                else:
                    merged[var_name].append(clean_line)

        output_parts = []

        if keep_non_variable_text:
            non_variable_chunks = self._get_non_variable_chunks(text, consumed_ranges)

            for chunk in non_variable_chunks:
                cleaned = chunk.strip()
                if cleaned:
                    output_parts.append(cleaned)

        for var_name, lines in merged.items():
            lines = self._strip_outer_blank_lines(lines)

            if add_pipe_separators:
                lines = self._add_pipe_separators_to_body_lines(lines)

            block = []
            block.append("${" + var_name + "={")
            block.extend(lines)
            block.append("}}")

            output_parts.append("\n".join(block))

        separator = "\n\n" if blank_line_between_blocks else "\n"
        result = separator.join(output_parts)

        return (result,)

    def _extract_variable_blocks(self, text: str):
        """
        Extracts blocks shaped like:

            ${name={
            body
            }}

        This is intentionally a small parser instead of only regex,
        because variable body text can contain braces.
        """
        blocks = []
        consumed_ranges = []

        pattern = re.compile(r"\$\{\s*([^={}]+?)\s*=\s*\{", re.MULTILINE)

        pos = 0

        while True:
            match = pattern.search(text, pos)
            if not match:
                break

            var_name = match.group(1).strip()

            body_start = match.end()
            i = body_start
            depth = 1

            while i < len(text):
                ch = text[i]

                if ch == "{":
                    depth += 1

                elif ch == "}":
                    depth -= 1

                    if depth == 0:
                        body_end = i
                        block_end = i + 1

                        # Folded prompt variable blocks usually close as "}}".
                        if block_end < len(text) and text[block_end] == "}":
                            block_end += 1

                        body = text[body_start:body_end]
                        blocks.append((var_name, body))
                        consumed_ranges.append((match.start(), block_end))

                        pos = block_end
                        break

                i += 1
            else:
                # Unclosed block. Stop safely.
                break

        return blocks, consumed_ranges

    def _get_non_variable_chunks(self, text: str, consumed_ranges):
        if not consumed_ranges:
            return [text]

        chunks = []
        last = 0

        for start, end in consumed_ranges:
            if start > last:
                chunks.append(text[last:start])
            last = end

        if last < len(text):
            chunks.append(text[last:])

        return chunks

    def _strip_outer_blank_lines(self, lines):
        start = 0
        end = len(lines)

        while start < end and lines[start].strip() == "":
            start += 1

        while end > start and lines[end - 1].strip() == "":
            end -= 1

        return lines[start:end]

    def _add_pipe_separators_to_body_lines(self, lines):
        """
        Adds | to every content line except the final content line.

        Skips:
        - blank lines
        - variable opener lines like ${x={
        - closing lines like }} or }
        - lines already ending in |
        """
        result = list(lines)

        content_indices = []

        for i, line in enumerate(result):
            stripped = line.strip()

            if not stripped:
                continue

            if re.match(r"^\s*\$\\?\{", line):
                continue

            if re.match(r"^\s*\}\}?\s*$", line):
                continue

            content_indices.append(i)

        if len(content_indices) <= 1:
            return result

        # Pipe every content line except the last content line.
        for i in content_indices[:-1]:
            line = result[i]

            if re.search(r"\|[ \t]*$", line):
                continue

            line_ending_match = re.search(r"(\r?\n)$", line)
            if line_ending_match:
                ending = line_ending_match.group(1)
                line_without_ending = line[:-len(ending)]
                result[i] = line_without_ending + "|" + ending
            else:
                result[i] = line + "|"

        return result

    def _add_pipe_separators_to_existing_text(self, text):
        """
        Applies a close equivalent of the requested regex replacement:

            find:
            (?m)^(?!\\s*\\$\\\\?\\{)(?!\\s*\\}\\}?)(?!.*\\|[ \\t]*$)([^\\r\\n]+)(\\r?\\n)(?!\\s*\\}\\}?)

            replace:
            \\1|\\2

        But applies it block-aware, so it won't add a pipe to the final item
        before a closing brace.
        """
        blocks, consumed_ranges = self._extract_variable_blocks(text)

        if not blocks:
            return self._pipe_regex_style(text)

        output = []
        last = 0

        for (var_name, body), (start, end) in zip(blocks, consumed_ranges):
            output.append(text[last:start])

            lines = body.splitlines()
            lines = self._add_pipe_separators_to_body_lines(lines)

            output.append("${" + var_name + "={")
            if lines:
                output.append("\n")
                output.append("\n".join(lines))
                output.append("\n")
            output.append("}}")

            last = end

        output.append(text[last:])

        return "".join(output)

    def _pipe_regex_style(self, text):
        """
        Direct regex fallback for plain text not parsed as variable blocks.
        """
        pattern = re.compile(
            r"(?m)^(?!\s*\$\\?\{)(?!\s*\}\}?)(?!.*\|[ \t]*$)([^\r\n]+)(\r?\n)(?!\s*\}\}?)"
        )

        return pattern.sub(r"\1|\2", text)


NODE_CLASS_MAPPINGS = {
    "CombineRepeatedVariables": CombineRepeatedVariables,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CombineRepeatedVariables": "Combine Repeated Variables",
}