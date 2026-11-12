import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_TYPE = "HunyuanVideoMGCache";
const STEPS = 50;
const HEADING = "mg_cache_heading";
const ESTIMATE = "mg_cache_estimate";
const RESULTS = "mg_cache_results";
const CONTROL_WIDTH = 320;
const MIN_WIDTH = 880;
const DEFAULT_SIZE = [960, 560];
const SECTIONS = [
    ["prompt", "01  提示词"],
    ["seed", "02  视频规格与采样"],
    ["alpha", "03  缓存与对照"],
];
const COLORS = { blank: "#4a5361", warmup: "#4786be", layer: "#4a5361", reused: "#1bb9a5", step: "#a98ae5" };
const POLICIES = { blank: "完整计算", warmup: "初始化层缓存", layer: "按层判断", step: "整步复用" };

function element(tag, parent, text, style) {
    const node = document.createElement(tag);
    if (text != null) node.textContent = text;
    if (style) Object.assign(node.style, style);
    parent.append(node);
    return node;
}

function addSectionHeadings(node) {
    for (const [firstWidget, label] of SECTIONS) {
        const index = node.widgets.findIndex(widget => widget.name === firstWidget);
        if (index < 0) continue;
        const heading = node.addCustomWidget({
            name: "mg_cache_heading_" + firstWidget,
            type: HEADING,
            options: { serialize: false },
            computeSize: () => [0, 19],
            draw(ctx, _node, width, y) {
                ctx.save();
                ctx.font = "600 12px sans-serif";
                ctx.textBaseline = "middle";
                ctx.fillStyle = "#8ed5e6";
                ctx.fillText(label, 15, y + 11);
                ctx.fillStyle = "#48515e";
                const lineStart = 25 + ctx.measureText(label).width;
                ctx.fillRect(lineStart, y + 11, Math.max(0, width - lineStart - 15), 1);
                ctx.restore();
            },
        });
        node.widgets.pop();
        node.widgets.splice(index, 0, heading);
    }
}

function resultUrl(report, key) {
    const query = new URLSearchParams({
        filename: report.files[key],
        subfolder: report.subfolder,
        type: "output",
    });
    return api.apiURL("/view?" + query.toString());
}

function renderVideo(parent, report, key, label) {
    const panel = element("section", parent, null, {
        minWidth: "0", background: "#191e26", padding: "8px", borderRadius: "6px",
    });
    const caption = element("div", panel, null, {
        display: "flex", justifyContent: "space-between", flexWrap: "wrap",
        gap: "2px 8px", marginBottom: "6px",
    });
    element("strong", caption, label);
    element("span", caption, "去噪 " + report[key].denoise_seconds.toFixed(2) + " 秒", { color: "#bac6d5" });
    const video = element("video", panel, null, {
        width: "100%", maxHeight: "clamp(84px, calc(100cqh - 80px), 360px)",
        aspectRatio: report.config.width + " / " + report.config.height,
        display: "block", objectFit: "contain", background: "#10141a", borderRadius: "4px",
    });
    video.controls = true;
    video.loop = true;
    video.preload = "metadata";
    video.src = resultUrl(report, key);
}

function chart(parent, label, series, unit) {
    const mean = series.mean === "inf" ? "∞" : series.mean.toFixed(3);
    element("strong", parent, label + " 均值 " + mean + unit);
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("viewBox", "0 0 340 104");
    svg.setAttribute("role", "img");
    svg.setAttribute("aria-label", label + " 逐帧曲线");
    svg.style.width = "100%";
    parent.append(svg);

    const finite = series.frames.filter(Number.isFinite);
    const low = finite.length ? Math.min(...finite) : 0;
    const observedHigh = finite.length ? Math.max(...finite) : 1;
    const spread = Math.max(observedHigh - low, 0.001);
    const high = observedHigh + spread * 0.1;
    const floor = low - spread * 0.1;
    const hasInfinity = finite.length < series.frames.length;

    function shape(tag, attributes, text, container = svg) {
        const item = document.createElementNS(svg.namespaceURI, tag);
        for (const [key, value] of Object.entries(attributes)) item.setAttribute(key, value);
        if (text != null) item.textContent = text;
        container.append(item);
        return item;
    }

    shape("path", { d: "M50 12V80H330", stroke: "currentColor", fill: "none", opacity: ".55" });
    shape("text", { x: 44, y: 14, "text-anchor": "end", fill: "currentColor", "font-size": 10 }, hasInfinity ? "∞" : high.toFixed(3));
    shape("text", { x: 44, y: 81, "text-anchor": "end", fill: "currentColor", "font-size": 10 }, floor.toFixed(3));
    shape("text", { x: 190, y: 100, "text-anchor": "middle", fill: "currentColor", "font-size": 10 }, "帧 1–" + series.frames.length);

    const points = series.frames.map((value, i) => {
        const x = series.frames.length === 1 ? 190 : 50 + 280 * i / (series.frames.length - 1);
        const finiteTop = hasInfinity ? 25 : 12;
        const y = value === "inf" ? 12 : 80 - (80 - finiteTop) * (value - floor) / (high - floor);
        return [x, y];
    });
    shape("polyline", { points: points.map(point => point.join(",")).join(" "), fill: "none", stroke: "#4ebbd4", "stroke-width": 1.7 });
    points.forEach(([cx, cy], i) => {
        const point = shape("circle", { cx, cy, r: 2.5, fill: "#4ebbd4" });
        const value = series.frames[i];
        shape("title", {}, "第 " + (i + 1) + " 帧 · " + (value === "inf" ? "∞" : value.toFixed(4)) + unit, point);
    });
}

function cacheMap(parent, report) {
    const steps = report.mg_cache.cache_steps;
    const layers = report.layer_count;
    const wholeSteps = steps.filter(item => item.policy === "step").length;
    const hits = steps.reduce((total, item) => total + item.reused_layers.length, 0);
    const section = element("details", parent);
    section.open = true;
    element("summary", section, "缓存决策 · 整步复用 " + wholeSteps + "/" + steps.length + " · 层注意力命中 " + hits + " 次");
    const legend = element("div", section, null, { display: "flex", flexWrap: "wrap", gap: "10px", margin: "6px 0" });
    for (const [kind, label] of [["blank", "完整计算"], ["warmup", "初始化"], ["reused", "复用注意力"], ["step", "整步复用"]]) {
        const item = element("span", legend, label);
        const swatch = element("span", item, null, {
            display: "inline-block", width: "10px", height: "10px", background: COLORS[kind], marginRight: "4px",
        });
        item.prepend(swatch);
    }
    element("div", section, "横轴 · 双流层 1–20、单流层 21–60　纵轴 · 去噪步 1–50");
    const canvas = element("canvas", section, null, { width: "100%", height: "210px", margin: "5px 0" });
    canvas.width = layers * 8;
    canvas.height = steps.length * 5;
    canvas.setAttribute("aria-label", "逐步逐层缓存决策热图");
    const ctx = canvas.getContext("2d");
    steps.forEach((item, row) => {
        const reused = new Set(item.reused_layers);
        for (let layer = 0; layer < layers; layer++) {
            ctx.fillStyle = COLORS[reused.has(layer) ? "reused" : item.policy];
            ctx.fillRect(layer * 8, row * 5, 7, 4);
        }
    });
    const picker = element("label", section, "查看第 ");
    const select = element("select", picker);
    for (const item of steps) {
        const option = element("option", select, item.step);
        option.value = item.step - 1;
    }
    picker.append(" 步");
    const description = element("div", section, null, { minHeight: "2em" });
    function describe(item) {
        description.textContent = "第 " + item.step + " 步 · " + POLICIES[item.policy] +
            (item.reused_layers.length ? " · 注意力复用层 " + item.reused_layers.map(i => i + 1).join("、") : "");
    }
    select.addEventListener("change", () => describe(steps[Number(select.value)]));
    canvas.addEventListener("mousemove", event => {
        const rect = canvas.getBoundingClientRect();
        const row = Math.floor((event.clientY - rect.top) / rect.height * steps.length);
        const layer = Math.floor((event.clientX - rect.left) / rect.width * layers);
        if (steps[row] && layer >= 0 && layer < layers) {
            const item = steps[row];
            const action = item.reused_layers.includes(layer) ? "复用注意力" :
                item.policy === "layer" ? "重新计算注意力" : POLICIES[item.policy];
            canvas.title = "第 " + item.step + " 步 · 第 " + (layer + 1) + " 层 · " + action;
        }
    });
    describe(steps[0]);
    element("p", section, "双流层命中时跳过 QKV 投影和注意力，单流层命中时仍执行联合投影。两类层均继续执行 MLP。");
}

function createResultTabs(parent) {
    const bar = element("div", parent, null, {
        display: "flex", gap: "6px", flexWrap: "wrap", margin: "0 0 10px",
    });
    const body = element("div", parent, null, {
        overflow: "auto", minHeight: "0", flex: "1", containerType: "size",
    });
    const status = element("div", body, null, {
        boxSizing: "border-box", height: "100%", minHeight: "110px", borderRadius: "6px",
        border: "1px dashed #455466", color: "#bac6d5",
    });
    status.setAttribute("role", "status");
    const empty = element("div", status, null, {
        display: "flex", height: "100%", padding: "0 12px", flexDirection: "column",
        alignItems: "center", justifyContent: "center", gap: "8px", textAlign: "center",
    });
    const icon = element("span", empty, "▶", {
        color: "#8ed5e6", background: "#273a48", borderRadius: "50%",
        width: "34px", lineHeight: "34px", paddingLeft: "2px",
    });
    icon.setAttribute("aria-hidden", "true");
    const messageText = element("span", empty);
    const tabs = new Map();

    function select(name) {
        for (const [key, { button, pane }] of tabs) {
            const selected = key === name;
            pane.hidden = !selected;
            button.setAttribute("aria-pressed", String(selected));
            button.style.background = selected ? "#205465" : "#323a46";
            button.style.borderColor = selected ? "#68bbd0" : "#48515e";
        }
        body.scrollTop = 0;
    }

    for (const [name, label] of [["videos", "视频预览"], ["quality", "质量曲线"], ["cache", "缓存热图"]]) {
        const button = element("button", bar, label, {
            color: "inherit", font: "inherit", padding: "3px 9px", cursor: "pointer",
            border: "1px solid #48515e", borderRadius: "4px",
        });
        button.type = "button";
        button.addEventListener("click", () => select(name));
        tabs.set(name, { button, pane: element("section", body) });
    }

    function clear(message) {
        messageText.textContent = message;
        status.hidden = false;
        for (const { button, pane } of tabs.values()) {
            button.hidden = true;
            pane.replaceChildren();
            pane.hidden = true;
        }
    }

    function render(report) {
        clear("");
        status.hidden = true;
        const { pane: videos } = tabs.get("videos");
        const videoStack = element("div", videos, null, {
            display: "flex", flexDirection: "column", gap: "10px",
        });
        if (report.baseline) renderVideo(videoStack, report, "baseline", "完整计算");
        renderVideo(videoStack, report, "mg_cache", "MG-Cache");
        const footer = element("div", videos, null, {
            display: "flex", gap: "12px", flexWrap: "wrap", marginTop: "5px",
        });
        if (report.baseline) {
            element("span", footer, "去噪加速比 " + report.speedup.toFixed(2) + "×");
            const { pane: quality } = tabs.get("quality");
            const curves = element("div", quality, null, {
                display: "flex", flexDirection: "column", gap: "10px",
            });
            chart(element("section", curves), "PSNR", report.quality.psnr, " dB");
            chart(element("section", curves), "SSIM", report.quality.ssim, "");
            tabs.get("quality").button.hidden = false;
        }
        cacheMap(tabs.get("cache").pane, report);
        const link = element("a", footer, "下载本次运行记录");
        link.href = resultUrl(report, "report");
        link.download = "report.json";
        link.style.color = "#82d4eb";
        tabs.get("videos").button.hidden = false;
        tabs.get("cache").button.hidden = false;
        select("videos");
    }

    return { clear, render };
}

function positionResultColumn(node, widget) {
    // ComfyUI applies the DOM margin to both axes. Compensate on Y while
    // shifting the preview past the native controls on X.
    const top = node.widgets[0].y;
    widget.y = top - CONTROL_WIDTH;
    widget.width = node.size[0] + CONTROL_WIDTH;
    widget.computedHeight = node.size[1] - top + CONTROL_WIDTH * 2;
}

app.registerExtension({
    name: "MGCache.HunyuanVideo",
    setup() {
        api.addEventListener("executing", event => {
            const node = app.graph?.getNodeById(event.detail);
            node?.mgCacheClear?.("生成中…");
        });
    },
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_TYPE) return;

        const created = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function (...args) {
            const result = created?.apply(this, args);
            addSectionHeadings(this);
            for (const widget of this.widgets) widget.width = CONTROL_WIDTH;
            for (const [name, minHeight, maxHeight] of [["prompt", 90, 128], ["negative_prompt", 72, 96]]) {
                const widget = this.widgets.find(item => item.name === name);
                widget.options.getMinHeight = () => minHeight;
                widget.options.getMaxHeight = () => maxHeight;
            }
            const estimate = document.createElement("div");
            Object.assign(estimate.style, {
                background: "#1c2934", color: "#c3d3e1", borderRadius: "6px",
                padding: "5px 8px", font: "12px/1.35 sans-serif", boxSizing: "border-box",
            });
            estimate.setAttribute("aria-live", "polite");
            const estimateWidget = this.addDOMWidget(ESTIMATE, ESTIMATE, estimate, {
                serialize: false, getMinHeight: () => 60, getMaxHeight: () => 60,
            });
            estimateWidget.width = CONTROL_WIDTH;

            const panel = document.createElement("div");
            Object.assign(panel.style, {
                boxSizing: "border-box", padding: "12px", background: "#222831", color: "#eef1f6",
                border: "1px solid #3b4857", borderRadius: "8px", overflow: "hidden", font: "13px/1.5 sans-serif",
                display: "flex", flexDirection: "column",
            });
            element("strong", panel, "04  运行结果", {
                color: "#8ed5e6", fontSize: "14px", paddingBottom: "8px", marginBottom: "8px",
                borderBottom: "1px solid #3b4857",
            });
            const results = createResultTabs(panel);
            const compare = this.widgets.find(widget => widget.name === "compare");
            const value = name => Number(this.widgets.find(widget => widget.name === name).value);
            const refreshEstimate = () => {
                const width = value("width");
                const height = value("height");
                const frames = value("frames");
                if (!(width >= 16 && height >= 16 && frames >= 1) || width % 16 || height % 16 || (frames - 1) % 4) {
                    estimate.textContent = "固定 50 步 · 宽高须为 16 的正倍数，帧数须为 4n + 1。";
                    return;
                }
                const textTokens = frames === 1 ? 315 : 256;
                const tokens = ((frames - 1) / 4 + 1) * width / 16 * height / 16 + textTokens;
                const gib = tokens * 3072 * 60 * 4 * (value("cfg_scale") > 1 ? 2 : 1) / 2 ** 30;
                estimate.textContent = "固定 " + STEPS + " 步 · 预计层缓存占用 " + gib.toFixed(2) + " GiB 锁页内存";
            };
            this.mgCacheClear = (message = "设置完成后，点击顶部「Run」排队运行。") => results.clear(message);
            this.mgCacheRefresh = refreshEstimate;
            for (const name of ["width", "height", "frames", "cfg_scale", "compare"]) {
                const widget = this.widgets.find(item => item.name === name);
                const callback = widget.callback;
                widget.callback = (...params) => {
                    callback?.apply(widget, params);
                    refreshEstimate();
                    this.mgCacheClear();
                };
            }

            this.mgCacheRender = report => {
                if (!report || Boolean(compare.value) !== report.config.compare) return;
                results.render(report);
            };
            const preview = this.addDOMWidget(RESULTS, RESULTS, panel, {
                serialize: false, margin: CONTROL_WIDTH + 10,
            });
            // The right column shares the height of the left column.
            preview.computeSize = () => [0, -4];
            preview.disabled = true;
            const layoutWidgets = this.layoutWidgets;
            this.layoutWidgets = function (...layoutArgs) {
                layoutWidgets.apply(this, layoutArgs);
                positionResultColumn(this, preview);
            };
            const computeSize = this.computeSize;
            this.computeSize = function (...sizeArgs) {
                const size = computeSize.apply(this, sizeArgs);
                size[0] = Math.max(size[0], MIN_WIDTH);
                return size;
            };
            refreshEstimate();
            this.mgCacheClear();
            this.setSize([...DEFAULT_SIZE]);
            return result;
        };

        // LiteGraph loads widget values by position. Keep headings out of the workflow format.
        const configure = nodeType.prototype.configure;
        nodeType.prototype.configure = function (data) {
            if (!Array.isArray(data.widgets_values)) return configure.call(this, data);
            const values = [];
            let index = 0;
            for (const widget of this.widgets) {
                if (index === data.widgets_values.length) break;
                values.push(widget.type === HEADING ? null : data.widgets_values[index++]);
            }
            return configure.call(this, { ...data, widgets_values: values });
        };

        const serialize = nodeType.prototype.onSerialize;
        nodeType.prototype.onSerialize = function (data) {
            serialize?.call(this, data);
            if (data.widgets_values) {
                data.widgets_values = data.widgets_values.filter((_, i) =>
                    ![HEADING, ESTIMATE, RESULTS].includes(this.widgets[i]?.type));
            }
        };

        const configured = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (...args) {
            configured?.apply(this, args);
            const [minWidth, minHeight] = this.computeSize();
            this.setSize([Math.max(this.size[0], minWidth), Math.max(this.size[1], minHeight)]);
            this.mgCacheRefresh?.();
        };

        const executed = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (output) {
            executed?.call(this, output);
            this.mgCacheRender?.(output.mg_cache_report?.[0]);
        };
    },
});
