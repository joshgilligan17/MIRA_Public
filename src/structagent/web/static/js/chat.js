/**
 * ChatUI - Handles chat interface functionality
 */
class ChatUI {
    /**
     * Create a ChatUI instance
     * @param {string} messages_div_id - ID of the messages container
     * @param {string} input_id - ID of the input field
     * @param {string} send_btn_id - ID of the send button
     */
    constructor(messages_div_id, input_id, send_btn_id) {
        this.messages_div = document.getElementById(messages_div_id);
        this.input = document.getElementById(input_id);
        this.send_btn = document.getElementById(send_btn_id);
        this.sendCallback = null;
        this._bindEvents();
    }

    _bindEvents() {
        const self = this;

        this.send_btn.addEventListener("click", function () {
            self._handleSend();
        });

        this.input.addEventListener("keypress", function (e) {
            if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                self._handleSend();
            }
        });
    }

    _handleSend() {
        const content = this.input.value.trim();
        if (content && this.sendCallback) {
            this.sendCallback(content);
            this.input.value = "";
        }
    }

    /**
     * Add a message to the chat
     * @param {string} role - "user", "assistant", or "system"
     * @param {string} content - Message content
     */
    addMessage(role, content) {
        const message_div = document.createElement("div");
        message_div.className = "message " + role;

        if (role === "tool-execution") {
            message_div.innerHTML = content;
        } else {
            message_div.textContent = content;
        }

        this.messages_div.appendChild(message_div);
        this._scrollToBottom();
    }

    /**
     * Add a tool execution indicator
     * @param {string} tool_name - Name of the tool being executed
     * @param {string} status - Initial status ("running", "success", "error")
     * @returns {HTMLElement} The created element for later updating
     */
    addToolExecution(tool_name, status) {
        const container = document.createElement("div");
        container.className = "message tool-execution";
        container.id = "tool-" + tool_name + "-" + Date.now();

        let spinnerHtml = "";
        if (status === "running") {
            spinnerHtml = '<div class="spinner"></div>';
        }

        container.innerHTML = `
            ${spinnerHtml}
            <span class="tool-name">${tool_name}</span>
            <span class="tool-status">${status}</span>
        `;

        this.messages_div.appendChild(container);
        this._scrollToBottom();
        return container;
    }

    /**
     * Update a tool execution with its result
     * @param {string} tool_name - Name of the tool
     * @param {string} result - Result or error message
     */
    setToolResult(tool_name, result) {
        // Find the most recent tool execution element for this tool
        const toolElements = this.messages_div.querySelectorAll(".tool-execution");
        let targetElement = null;

        for (let i = toolElements.length - 1; i >= 0; i--) {
            const el = toolElements[i];
            if (el.querySelector(".tool-name").textContent === tool_name) {
                targetElement = el;
                break;
            }
        }

        if (targetElement) {
            targetElement.querySelector(".spinner").remove();
            targetElement.querySelector(".tool-status").textContent = "completed";

            const resultDiv = document.createElement("div");
            resultDiv.className = "tool-result";
            resultDiv.textContent = result;
            targetElement.appendChild(resultDiv);

            this._scrollToBottom();
        }
    }

    /**
     * Clear all messages
     */
    clearMessages() {
        this.messages_div.innerHTML = "";
    }

    /**
     * Set the callback for send events
     * @param {Function} callback - Function to call when user sends a message
     */
    onSend(callback) {
        this.sendCallback = callback;
    }

    /**
     * Enable or disable the input and send button
     * @param {boolean} enabled - Whether inputs should be enabled
     */
    setEnabled(enabled) {
        this.input.disabled = !enabled;
        this.send_btn.disabled = !enabled;
    }

    _scrollToBottom() {
        this.messages_div.scrollTop = this.messages_div.scrollHeight;
    }
}
