/**
 * PDBViewer - Wrapper around 3Dmol.js for viewing protein structures
 */
class PDBViewer {
    /**
     * Create a PDBViewer instance
     * @param {string} container_id - The ID of the container element
     */
    constructor(container_id) {
        this.container_id = container_id;
        this.container = document.getElementById(container_id);
        this.viewer = null;
        this.hoverCallback = null;
        this._initViewer();
    }

    _initViewer() {
        this.viewer = $3Dmol.createViewer(this.container_id, {
            backgroundColor: "black"
        });

        const self = this;
        this.viewer.mousehover(function (atom) {
            if (self.hoverCallback && atom) {
                const info = {
                    resn: atom.resn,
                    resi: atom.resi,
                    chain: atom.chain,
                    elem: atom.elem,
                    x: atom.x,
                    y: atom.y,
                    z: atom.z,
                    b: atom.b
                };
                self.hoverCallback(info);
            }
        });
    }

    /**
     * Load a structure from the RCSB PDB
     * @param {string} pdbId - The PDB ID to load
     */
    loadStructure(pdbId) {
        const loader = this;
        $.get("https://files.rcsb.org/download/" + pdbId.toUpperCase() + ".pdb", function (data) {
            loader.viewer.addModel(data, "pdb");
            loader._applyDefaultStyle();
            loader.viewer.zoomTo();
            loader.viewer.render();
        }).fail(function () {
            console.error("Failed to load PDB structure:", pdbId);
        });
    }

    _applyDefaultStyle() {
        this.viewer.setStyle({}, {
            cartoon: {
                color: "spectrum"
            }
        });
    }

    /**
     * Highlight specific residues with a given color
     * @param {string} chain - Chain identifier
     * @param {number[]} residues - Array of residue indices
     * @param {string} color - CSS color string
     */
    highlightResidues(chain, residues, color) {
        const sel = {};
        sel.chain = chain;
        sel.resi = residues;

        this.viewer.addStyle(sel, {
            stick: {
                colorscheme: color
            }
        });
        this.viewer.render();
    }

    /**
     * Color structure by secondary structure elements
     * @param {Array} elements - Array of {range: [start, end], type: "helix"|"strand"|"coil"} objects
     */
    colorBySecondaryStructure(elements) {
        const colorMap = {
            "helix": "red",
            "strand": "yellow",
            "coil": "gray"
        };

        // First reset all styles
        this.viewer.setStyle({}, {
            cartoon: {color: "gray"}
        });

        // Apply colors per element
        for (const el of elements) {
            const [start, end] = el.range;
            const color = colorMap[el.type] || "gray";

            this.viewer.addStyle({
                resi: `${start}-${end}`
            }, {
                cartoon: {color: color}
            });
        }

        this.viewer.render();
    }

    /**
     * Show interface between two sets of residues
     * @param {Object} chainA_residues - {chain: string, residues: number[]}
     * @param {Object} chainB_residues - {chain: string, residues: number[]}
     */
    showInterface(chainA_residues, chainB_residues) {
        // Show chain A residues in yellow
        this.viewer.addStyle({
            chain: chainA_residues.chain,
            resi: chainA_residues.residues
        }, {
            cartoon: {color: "yellow"}
        });

        // Show chain B residues in cyan
        this.viewer.addStyle({
            chain: chainB_residues.chain,
            resi: chainB_residues.residues
        }, {
            cartoon: {color: "cyan"}
        });

        this.viewer.render();
    }

    /**
     * Highlight outliers in red
     * @param {number[]} residues - Array of residue indices to highlight
     */
    highlightOutliers(residues) {
        this.viewer.addStyle({
            resi: residues
        }, {
            stick: {
                colorscheme: "red"
            }
        });
        this.viewer.render();
    }

    /**
     * Reset the view to default styling
     */
    resetView() {
        this.viewer.removeAllModels();
        this.viewer.render();
    }

    /**
     * Set callback for hover events
     * @param {Function} callback - Function to call on hover with atom info
     */
    setHoverCallback(callback) {
        this.hoverCallback = callback;
    }
}
