import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "custom_nodes4macos.DreamFactory",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "FusionMLXDreamFactory") {
            const origOnNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                if (origOnNodeCreated) {
                    origOnNodeCreated.apply(this, arguments);
                }
                this.addWidget("button", "🎬 打开梦工厂", null, () => {
                    const base = window.location.origin;
                    const htmlPath = base + "/extensions/custom_nodes4macos/dream_factory.html";
                    window.open(htmlPath, "_blank", "width=900,height=800");
                });
            };
        }
    },
});
