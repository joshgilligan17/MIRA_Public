/**
 * MIRA Web Application
 */

(function () {
    // Application state
    let currentPdbId = null;
    let ws = null;
    let chatUI = null;
    let pdbViewer = null;

    // WebSocket URL
    const WS_URL = "ws://localhost:8000/ws";

    /**
     * Initialize the application
     */
    function init() {
        // Initialize UI components
        chatUI = new ChatUI("chat-messages", "chat-input", "send-btn");
        pdbViewer = new PDBViewer("viewer-container");

        // Set up chat send handler
        chatUI.onSend(handleSendMessage);

        // Set up PDB load handler
        document.getElementById("load-pdb-btn").addEventListener("click", handleLoadPdb);
        document.getElementById("pdb-id-input").addEventListener("keypress", function (e) {
            if (e.key === "Enter") {
                handleLoadPdb();
            }
        });

        // Set up hover callback for info panel
        pdbViewer.setHoverCallback(handleHover);

        // Enable input
        chatUI.setEnabled(true);

        // Connect to WebSocket
        connectWebSocket();
    }

    /**
     * Connect to the WebSocket server
     */
    function connectWebSocket() {
        ws = new WebSocket(WS_URL);

        ws.onopen = function () {
            console.log("WebSocket connected");
            chatUI.addMessage("assistant", "Connected to MIRA server. Enter a PDB ID to begin.");
        };

        ws.onmessage = function (event) {
            handleServerMessage(JSON.parse(event.data));
        };

        ws.onerror = function (error) {
            console.error("WebSocket error:", error);
            chatUI.addMessage("assistant", "WebSocket error occurred. Please check if the server is running.");
        };

        ws.onclose = function () {
            console.log("WebSocket disconnected");
            chatUI.addMessage("assistant", "Disconnected from server. Attempting to reconnect...");
            setTimeout(connectWebSocket, 3000);
        };
    }

    /**
     * Handle messages from the server
     * @param {Object} message - Parsed message object
     */
    function handleServerMessage(message) {
        switch (message.type) {
            case "chat_response":
                handleChatResponse(message);
                break;

            case "tool_execution":
                handleToolExecution(message);
                break;

            case "viewer_update":
                handleViewerUpdate(message);
                break;

            case "error":
                handleError(message);
                break;

            default:
                console.warn("Unknown message type:", message.type);
        }
    }

    /**
     * Handle chat response from server
     * @param {Object} message
     */
    function handleChatResponse(message) {
        chatUI.addMessage("assistant", message.content);
    }

    /**
     * Handle tool execution status
     * @param {Object} message
     */
    function handleToolExecution(message) {
        if (message.status === "started") {
            chatUI.addToolExecution(message.tool_name, "running");
        } else if (message.status === "completed") {
            chatUI.setToolResult(message.tool_name, message.result || "Completed");
        } else if (message.status === "error") {
            chatUI.setToolResult(message.tool_name, "Error: " + message.error);
        }
    }

    /**
     * Handle viewer update commands
     * @param {Object} message
     */
    function handleViewerUpdate(message) {
        const { action, data } = message;

        switch (action) {
            case "load":
                if (data.pdb_id) {
                    pdbViewer.loadStructure(data.pdb_id);
                }
                break;

            case "color_ss":
                if (data.elements) {
                    pdbViewer.colorBySecondaryStructure(data.elements);
                }
                break;

            case "highlight_residues":
                if (data.chain && data.residues && data.color) {
                    pdbViewer.highlightResidues(data.chain, data.residues, data.color);
                }
                break;

            case "show_interface":
                if (data.chainA_residues && data.chainB_residues) {
                    pdbViewer.showInterface(data.chainA_residues, data.chainB_residues);
                }
                break;

            case "highlight_outliers":
                if (data.residues) {
                    pdbViewer.highlightOutliers(data.residues);
                }
                break;

            default:
                console.warn("Unknown viewer action:", action);
        }
    }

    /**
     * Handle errors from server
     * @param {Object} message
     */
    function handleError(message) {
        chatUI.addMessage("assistant", "Error: " + message.content);
    }

    /**
     * Handle sending a chat message
     * @param {string} content - Message content
     */
    function handleSendMessage(content) {
        if (!ws || ws.readyState !== WebSocket.OPEN) {
            chatUI.addMessage("assistant", "Not connected to server. Please wait...");
            return;
        }

        // Add user message to chat
        chatUI.addMessage("user", content);

        // Send to server
        ws.send(JSON.stringify({
            type: "chat_message",
            content: content,
            pdb_id: currentPdbId
        }));
    }

    /**
     * Handle PDB ID load request
     */
    function handleLoadPdb() {
        const input = document.getElementById("pdb-id-input");
        const pdbId = input.value.trim().toLowerCase();

        if (!pdbId) {
            return;
        }

        currentPdbId = pdbId;
        pdbViewer.loadStructure(pdbId);

        // Notify server if connected
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({
                type: "viewer_action",
                action: "load",
                pdb_id: pdbId
            }));
        }
    }

    /**
     * Handle hover events on the viewer
     * @param {Object} info - Atom information
     */
    function handleHover(info) {
        const infoPanel = document.getElementById("info-panel");
        if (info && info.resn) {
            infoPanel.textContent = `Residue: ${info.resn}${info.resi} | Chain: ${info.chain} | Element: ${info.elem} | B-factor: ${info.b ? info.b.toFixed(2) : "N/A"}`;
        } else {
            infoPanel.textContent = "";
        }
    }

    // Initialize when DOM is ready
    document.addEventListener("DOMContentLoaded", init);
})();
