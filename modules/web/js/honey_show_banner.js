import { app } from "../../../scripts/app.js";


app.registerExtension({
    name: "Honey.ShowBanner",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "HoneyShowBanner") {
            return;
        }

        const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
        const originalOnExecuted = nodeType.prototype.onExecuted;
        const originalOnPropertyChanged = nodeType.prototype.onPropertyChanged;

        nodeType.prototype.onNodeCreated = function () {
            originalOnNodeCreated?.apply(this, arguments);

            this.properties ??= {};

            addPropertyIfMissing(this, "banner_text", "Waiting for input…", "string");
            addPropertyIfMissing(this, "font_size", 16, "number");
            addPropertyIfMissing(this, "bold", true, "boolean");
            addPropertyIfMissing(this, "text_color", "#ffffff", "string");
            addPropertyIfMissing(this, "background_color", "#000000", "string");
            addPropertyIfMissing(this, "border_color", "#555555", "string");
            addPropertyIfMissing(this, "padding", 12, "number");
            addPropertyIfMissing(this, "line_height", 21, "number");
            addPropertyIfMissing(this, "wrap_text", true, "boolean");
            addPropertyIfMissing(
                this,
                "text_align",
                "center",
                "enum",
                {
                    values: [
                        "left",
                        "center",
                        "right",
                    ],
                }
            );
            addPropertyIfMissing(this, "max_lines", 0, "number");

            /*
             * Give newly created nodes a useful initial size.
             * This runs only when the node is created.
             * It does not resize the node during execution.
             */
            if (!this.flags?.collapsed) {
                this.setSize([
                    Math.max(this.size[0], 320),
                    Math.max(this.size[1], 130),
                ]);
            }

            const bannerWidget = {
                name: "honey_banner",
                type: "honey_banner",
                value: this.properties.banner_text,

                draw: function (ctx, node, width, y, height) {
                    drawBanner(ctx, node, width, y, height);
                },

                /*
                 * This provides a minimum height but does not change
                 * the node size after each execution.
                 */
                computeSize: function (width) {
                    return [width, 70];
                },

                serializeValue: function () {
                    return undefined;
                },

                mouse: function () {
                    return false;
                },
            };

            this.addCustomWidget(bannerWidget);
            this.honeyBannerWidget = bannerWidget;
        };

        nodeType.prototype.onExecuted = function (message) {
            originalOnExecuted?.apply(this, arguments);

            let received = message?.banner_text;

            if (Array.isArray(received)) {
                received = received.join("\n");
            }

            if (received === undefined || received === null) {
                received = "";
            }

            const text = String(received);

            this.properties ??= {};
            this.properties.banner_text = text;

            if (this.honeyBannerWidget) {
                this.honeyBannerWidget.value = text;
            }

            /*
             * Intentionally do not call setSize() here.
             * The node retains its current manually selected dimensions.
             */
            this.setDirtyCanvas(true, true);
        };

        nodeType.prototype.onPropertyChanged = function (
            propertyName,
            value,
            previousValue
        ) {
            const result = originalOnPropertyChanged?.apply(
                this,
                arguments
            );

            if (propertyName === "banner_text") {
                this.honeyBannerWidget.value = String(value ?? "");
            }

            this.setDirtyCanvas(true, true);

            return result;
        };
    },
});


function addPropertyIfMissing(
    node,
    name,
    defaultValue,
    type,
    options = undefined
) {
    if (node.properties[name] !== undefined) {
        return;
    }

    if (typeof node.addProperty === "function") {
        node.addProperty(
            name,
            defaultValue,
            type,
            options
        );
    } else {
        node.properties[name] = defaultValue;
    }
}


function drawBanner(ctx, node, width, y, height) {
    const properties = node.properties ?? {};

    const padding = clampNumber(
        properties.padding,
        0,
        100,
        12
    );

    const fontSize = clampNumber(
        properties.font_size,
        6,
        200,
        16
    );

    const lineHeight = clampNumber(
        properties.line_height,
        fontSize,
        300,
        Math.ceil(fontSize * 1.3)
    );

    const maxLines = clampNumber(
        properties.max_lines,
        0,
        1000,
        0
    );

    const bold = properties.bold !== false;
    const wrapText = properties.wrap_text !== false;

    const textAlign = [
        "left",
        "center",
        "right",
    ].includes(properties.text_align)
        ? properties.text_align
        : "center";

    const backgroundColor = validColor(
        properties.background_color,
        "#000000"
    );

    const textColor = validColor(
        properties.text_color,
        "#ffffff"
    );

    const borderColor = validColor(
        properties.border_color,
        "#555555"
    );

    const boxX = 6;
    const boxY = y + 2;
    const boxWidth = Math.max(1, width - 12);
    const boxHeight = Math.max(46, height - 4);

    ctx.save();

    ctx.fillStyle = backgroundColor;
    ctx.beginPath();

    if (ctx.roundRect) {
        ctx.roundRect(
            boxX,
            boxY,
            boxWidth,
            boxHeight,
            6
        );
    } else {
        ctx.rect(
            boxX,
            boxY,
            boxWidth,
            boxHeight
        );
    }

    ctx.fill();

    ctx.strokeStyle = borderColor;
    ctx.lineWidth = 1;
    ctx.stroke();

    ctx.fillStyle = textColor;
    ctx.font =
        `${bold ? "bold " : ""}${fontSize}px Arial, sans-serif`;

    ctx.textBaseline = "top";
    ctx.textAlign = textAlign;

    const text = String(
        properties.banner_text ?? ""
    );

    const innerX = boxX + padding;
    const innerY = boxY + padding;
    const innerWidth = Math.max(
        1,
        boxWidth - padding * 2
    );
    const innerHeight = Math.max(
        1,
        boxHeight - padding * 2
    );

    let drawX;

    if (textAlign === "left") {
        drawX = innerX;
    } else if (textAlign === "right") {
        drawX = innerX + innerWidth;
    } else {
        drawX = innerX + innerWidth / 2;
    }

    drawTextInsideBox(
        ctx,
        text,
        drawX,
        innerY,
        innerWidth,
        innerHeight,
        lineHeight,
        wrapText,
        maxLines
    );

    ctx.restore();
}


function drawTextInsideBox(
    ctx,
    text,
    x,
    y,
    maxWidth,
    maxHeight,
    lineHeight,
    wrapText,
    configuredMaxLines
) {
    const heightLimitedLines = Math.max(
        1,
        Math.floor(maxHeight / lineHeight)
    );

    const allowedLines =
        configuredMaxLines > 0
            ? Math.min(configuredMaxLines, heightLimitedLines)
            : heightLimitedLines;

    const sourceLines = String(text).split(/\r?\n/);
    const renderedLines = [];

    for (const sourceLine of sourceLines) {
        if (renderedLines.length >= allowedLines) {
            break;
        }

        if (!wrapText) {
            renderedLines.push(sourceLine);
            continue;
        }

        const wrapped = wrapLine(
            ctx,
            sourceLine,
            maxWidth
        );

        for (const line of wrapped) {
            if (renderedLines.length >= allowedLines) {
                break;
            }

            renderedLines.push(line);
        }
    }

    const contentWasTruncated = hasMoreContent(
        ctx,
        sourceLines,
        renderedLines,
        maxWidth,
        wrapText
    );

    if (contentWasTruncated && renderedLines.length > 0) {
        renderedLines[renderedLines.length - 1] =
            addEllipsis(
                ctx,
                renderedLines[renderedLines.length - 1],
                maxWidth
            );
    }

    renderedLines.forEach((line, index) => {
        ctx.fillText(
            line,
            x,
            y + index * lineHeight
        );
    });
}


function wrapLine(ctx, line, maxWidth) {
    if (line === "") {
        return [""];
    }

    const words = line.split(/\s+/);
    const lines = [];
    let currentLine = "";

    for (const word of words) {
        const candidate = currentLine
            ? `${currentLine} ${word}`
            : word;

        if (
            ctx.measureText(candidate).width <= maxWidth
            || currentLine === ""
        ) {
            currentLine = candidate;
        } else {
            lines.push(currentLine);
            currentLine = word;
        }
    }

    if (currentLine !== "") {
        lines.push(currentLine);
    }

    return lines;
}


function addEllipsis(ctx, line, maxWidth) {
    const ellipsis = "…";
    let result = String(line);

    while (
        result.length > 0
        && ctx.measureText(result + ellipsis).width > maxWidth
    ) {
        result = result.slice(0, -1);
    }

    return result + ellipsis;
}


function hasMoreContent(
    ctx,
    sourceLines,
    renderedLines,
    maxWidth,
    wrapText
) {
    let totalRequiredLines = 0;

    for (const line of sourceLines) {
        totalRequiredLines += wrapText
            ? wrapLine(ctx, line, maxWidth).length
            : 1;
    }

    return totalRequiredLines > renderedLines.length;
}


function clampNumber(
    value,
    minimum,
    maximum,
    fallback
) {
    const number = Number(value);

    if (!Number.isFinite(number)) {
        return fallback;
    }

    return Math.min(
        maximum,
        Math.max(minimum, number)
    );
}


function validColor(value, fallback) {
    const text = String(value ?? "").trim();

    if (
        /^#[0-9a-fA-F]{3}$/.test(text)
        || /^#[0-9a-fA-F]{6}$/.test(text)
        || /^#[0-9a-fA-F]{8}$/.test(text)
        || /^rgb/i.test(text)
        || /^hsl/i.test(text)
    ) {
        return text;
    }

    return fallback;
}