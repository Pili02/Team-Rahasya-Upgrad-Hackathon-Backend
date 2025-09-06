// Minimal frontend-only app to render a radial mind map using D3 v7

// -------------------------------
// API layer (real or mock)
// -------------------------------
const BACKEND_URL = 'http://localhost:8000/api/v1/generate_mindmap';

// Fixed mock JSON (from your sample)
const FIXED_MOCK_RESPONSE = {
    "root": "Mastering Generative AI",
    "nodes": [
        {
            "id": 1,
            "title": "Foundations of Generative AI",
            "description": "Understand the fundamental concepts of generative AI by exploring types such as Generative Adversarial Networks (GANs), Variational Autoencoders (VAEs), and Normalizing Flows. Learn about their applications in areas like image synthesis, data augmentation, and text generation. Additionally, delve into the limitations of these models, including mode collapse, vanishing gradients, and the need for careful regularization.",
            "resources": [
                "https://www.coursera.org/learn/generative-ai-foundations-and-concepts",
                "http://bactra.org/weblog/statsgen-f25.html"
            ],
            "children": [
                {
                    "id": 5,
                    "title": "Generative Adversarial Networks (GANs)",
                    "description": "Learn about GANs, a type of generative model that uses a generator and discriminator to generate new data samples. Explore their applications in image synthesis, data augmentation, and more.",
                    "resources": [
                        "https://www.xenonstack.com/blog/gans-for-image-synthesis",
                        "https://en.wikipedia.org/wiki/Generative_adversarial_network"
                    ],
                    "children": []
                },
                {
                    "id": 6,
                    "title": "Variational Autoencoders (VAEs)",
                    "description": "Discover VAEs, a type of generative model that uses an encoder and decoder to learn probabilistic representations of data. Learn about their applications in image compression, dimensionality reduction, and more.",
                    "resources": [
                        "https://www.linkedin.com/pulse/from-compression-creation-variational-autoencoders-vaes-debasish-deb-cxhif",
                        "https://dilipkumar.medium.com/variational-autoencoders-vaes-modles-597b1a71f97b"
                    ],
                    "children": []
                },
                {
                    "id": 7,
                    "title": "Normalizing Flows",
                    "description": "Explore normalizing flows, a type of generative model that uses invertible transformations to learn complex distributions. Learn about their applications in image synthesis, data augmentation, and more.",
                    "resources": [
                        "https://www.researchgate.net/publication/341222454_Normalizing_Flows_An_Introduction_and_Review_of_Current_Methods",
                        "https://deepgenerativemodels.github.io/notes/flow/",
                        "https://machinelearning.apple.com/research/normalizing-flows"
                    ],
                    "children": []
                },
                {
                    "id": 8,
                    "title": "Image Synthesis",
                    "description": "Learn how generative AI models like GANs and VAEs can be used to generate new images that resemble existing ones. Explore applications in computer vision, art, and more.",
                    "resources": [],
                    "children": []
                },
                {
                    "id": 9,
                    "title": "Data Augmentation",
                    "description": "Discover how generative AI models like GANs and VAEs can be used to generate new data samples that augment existing datasets. Learn about applications in machine learning, computer vision, and more.",
                    "resources": [
                        "https://aws.amazon.com/what-is/data-augmentation/",
                        "https://addepto.com/blog/generative-ai-for-data-augmentation-how-to-use-it/"
                    ],
                    "children": []
                },
                {
                    "id": 10,
                    "title": "Text Generation",
                    "description": "Learn how generative AI models like GANs and VAEs can be used to generate new text samples that resemble existing ones. Explore applications in natural language processing, chatbots, and more.",
                    "resources": [],
                    "children": []
                },
                {
                    "id": 11,
                    "title": "Mode Collapse",
                    "description": "Understand the concept of mode collapse in generative AI models like GANs and VAEs, where the generated samples lack diversity. Learn about its implications and potential solutions.",
                    "resources": [
                        "https://en.wikipedia.org/wiki/Mode_collapse"
                    ],
                    "children": []
                },
                {
                    "id": 12,
                    "title": "Vanishing Gradients",
                    "description": "Learn about vanishing gradients in generative AI models like GANs and VAEs, where the gradients used to update model parameters become increasingly small. Understand its implications and potential solutions.",
                    "resources": [
                        "https://vanishinggradients.fireside.fm/"
                    ],
                    "children": []
                },
                {
                    "id": 13,
                    "title": "Regularization Techniques",
                    "description": "Discover various regularization techniques that can be used to improve the performance of generative AI models like GANs and VAEs. Learn about their applications and potential benefits.",
                    "resources": [],
                    "children": []
                }
            ]
        },
        {
            "id": 2,
            "title": "Core Components of Generative AI",
            "description": "Learn about the key building blocks of generative models, including neural networks, optimization techniques, and data preprocessing.",
            "resources": [
                "https://www.surfsidemedia.in/post/what-are-the-key-components-of-a-generative-ai-model",
                "https://arbisoft.com/blogs/generative-ai-fundamentals-understanding-generative-models"
            ],
            "children": []
        },
        {
            "id": 3,
            "title": "Practical Application of Generative AI",
            "description": "Develop hands-on experience with generative models using popular libraries (e.g., TensorFlow, PyTorch) and frameworks (e.g., Keras).",
            "resources": [],
            "children": []
        },
        {
            "id": 4,
            "title": "Best Practices for Working with Generative AI",
            "description": "Learn common pitfalls, debugging techniques, and optimization strategies for working effectively with generative models.",
            "resources": [
                "https://www.heliosz.ai/blogs/7-best-practices-for-gen-ai-implementation/",
                "https://ardas-it.com/generative-ai-architecture"
            ],
            "children": []
        }
    ],
    "total_nodes": 13,
    "generated_at": "2025-09-06T11:07:18.893091"
};

async function fetchMindMap(promptText) {
    if (BACKEND_URL) {
        const res = await fetch(BACKEND_URL, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ description: promptText })
        });
        if (!res.ok) throw new Error(await res.text().catch(() => `HTTP ${res.status}`));
        return res.json();
    }
    // fallback to mock during development: ALWAYS return the fixed sample
    // (you can still paste full JSON in the input to override)
    await new Promise(r => setTimeout(r, 400));
    return JSON.parse(JSON.stringify(FIXED_MOCK_RESPONSE));
}

// -------------------------------
// Mock API (dev only)
// -------------------------------
async function fetchMindMapFromMockAPI(promptText) {
    // Simulate latency
    await new Promise(r => setTimeout(r, 650));

    // If promptText includes photosynthesis, generate a themed mock; else return provided example.
    const lower = (promptText || '').toLowerCase();
    if (lower.includes('photo')) {
        return getPhotosynthesisMock();
    }
    return getDSAExampleMock();
}

function getDSAExampleMock() {
    return {
        "root": "Mastering Data Structures and Algorithms",
        "nodes": [
            {
                "id": 1,
                "title": "Understand Fundamentals",
                "description": "Learn basic concepts of algorithms and data structures, including Big-O notation, recursion, and common problems.",
                "time_left": "2-3 weeks",
                "difficulty": "Beginner",
                "resources": [],
                "prerequisites": [],
                "children": [
                    {
                        "id": 2,
                        "title": "Basic Data Structures",
                        "description": "Study arrays, linked lists, stacks, and queues.",
                        "time_left": "1-2 weeks",
                        "difficulty": "Beginner",
                        "resources": [],
                        "prerequisites": [1],
                        "children": [],
                        "metadata": {}
                    },
                    {
                        "id": 3,
                        "title": "Algorithms Fundamentals",
                        "description": "Learn about sorting, searching, and graph algorithms.",
                        "time_left": "2-3 weeks",
                        "difficulty": "Beginner",
                        "resources": [],
                        "prerequisites": [1],
                        "children": [],
                        "metadata": {}
                    }
                ],
                "metadata": {}
            },
            {
                "id": 4,
                "title": "Competitive Programming",
                "description": "Learn to solve problems on platforms like LeetCode, HackerRank, and CodeForces.",
                "time_left": "2-3 weeks",
                "difficulty": "Intermediate",
                "resources": [],
                "prerequisites": [1],
                "children": [
                    {
                        "id": 5,
                        "title": "Problem-Solving Strategies",
                        "description": "Learn to approach problems systematically, using techniques like divide and conquer, dynamic programming, and greedy algorithms.",
                        "time_left": "1-2 weeks",
                        "difficulty": "Intermediate",
                        "resources": [],
                        "prerequisites": [4],
                        "children": [],
                        "metadata": {}
                    },
                    {
                        "id": 6,
                        "title": "Practice and Review",
                        "description": "Solve problems regularly, reviewing concepts and practicing different types of problems.",
                        "time_left": "2-3 weeks",
                        "difficulty": "Beginner",
                        "resources": [],
                        "prerequisites": [4],
                        "children": [],
                        "metadata": {}
                    }
                ],
                "metadata": {}
            },
            {
                "id": 7,
                "title": "Advanced Topics",
                "description": "Explore advanced topics like trees, graphs, and hash tables.",
                "time_left": "2-3 weeks",
                "difficulty": "Advanced",
                "resources": [],
                "prerequisites": [1],
                "children": [
                    {
                        "id": 8,
                        "title": "Trees and Graphs",
                        "description": "Study basic tree and graph algorithms, including traversal methods and shortest paths.",
                        "time_left": "1-2 weeks",
                        "difficulty": "Advanced",
                        "resources": [],
                        "prerequisites": [7],
                        "children": [],
                        "metadata": {}
                    },
                    {
                        "id": 9,
                        "title": "Hash Tables and Bit Manipulation",
                        "description": "Learn about hash tables, bit manipulation, and bitwise operations.",
                        "time_left": "1-2 weeks",
                        "difficulty": "Advanced",
                        "resources": [],
                        "prerequisites": [7],
                        "children": [],
                        "metadata": {}
                    }
                ],
                "metadata": {}
            }
        ],
        "total_nodes": 9,
        "estimated_total_time": "4.3 months",
        "complexity_score": 0.55,
        "generated_at": new Date().toISOString()
    };
}

function getPhotosynthesisMock() {
    return {
        root: "Photosynthesis",
        nodes: [
            { id: 1, title: "Overview", description: "Conversion of light energy to chemical energy.", time_left: "~1 week", difficulty: "Beginner", resources: [], prerequisites: [], children: [
                { id: 2, title: "Equation", description: "6CO2 + 6H2O + light → C6H12O6 + 6O2", time_left: "2 days", difficulty: "Beginner", resources: [], prerequisites: [1], children: [], metadata: {}},
                { id: 3, title: "Chloroplast", description: "Thylakoids, grana, stroma", time_left: "3 days", difficulty: "Beginner", resources: [], prerequisites: [1], children: [], metadata: {}},
            ], metadata: {}},
            { id: 4, title: "Light Reactions", description: "Happen in thylakoid membranes, produce ATP/NADPH", time_left: "1 week", difficulty: "Intermediate", resources: [], prerequisites: [1], children: [
                { id: 5, title: "Photosystems", description: "PSII and PSI", time_left: "2-3 days", difficulty: "Intermediate", resources: [], prerequisites: [4], children: [], metadata: {}},
                { id: 6, title: "Electron Transport", description: "Creates proton gradient for ATP synthase", time_left: "3-4 days", difficulty: "Intermediate", resources: [], prerequisites: [4], children: [], metadata: {}},
            ], metadata: {}},
            { id: 7, title: "Calvin Cycle", description: "Carbon fixation in stroma to form sugars", time_left: "1 week", difficulty: "Intermediate", resources: [], prerequisites: [1], children: [
                { id: 8, title: "Phases", description: "Carboxylation, reduction, regeneration", time_left: "3-4 days", difficulty: "Intermediate", resources: [], prerequisites: [7], children: [], metadata: {}},
                { id: 9, title: "Rubisco", description: "Key enzyme that fixes CO2", time_left: "2-3 days", difficulty: "Intermediate", resources: [], prerequisites: [7], children: [], metadata: {}},
            ], metadata: {}},
        ],
        total_nodes: 9,
        estimated_total_time: "3-4 weeks",
        complexity_score: 0.42,
        generated_at: new Date().toISOString(),
    };
}

// -------------------------------
// Transform JSON to hierarchy
// -------------------------------
function toHierarchy(data) {
    const safe = data || { root: 'Mind Map', nodes: [] };
    const nodes = Array.isArray(safe.nodes) ? safe.nodes : [];
    const root = { id: 'root', title: safe.root || 'Mind Map', children: [] };
    
    // Build proper parent-child relationships
    const nodeMap = new Map();
    nodes.forEach(n => {
        n.children = Array.isArray(n.children) ? n.children : [];
        n.prerequisites = Array.isArray(n.prerequisites) ? n.prerequisites : [];
        nodeMap.set(n.id, n);
    });
    
    // Only root-level nodes (no prerequisites) go under root
    root.children = nodes.filter(n => (n.prerequisites || []).length === 0);
    
    return root;
}

// -------------------------------
// Mind map renderer (radial layout)
// -------------------------------
const svg = d3.select('#mindmap');
const gRoot = svg.append('g').attr('class', 'g-root');
const gLinks = gRoot.append('g').attr('class', 'g-links');
const gNodes = gRoot.append('g').attr('class', 'g-nodes');

const zoom = d3.zoom().scaleExtent([0.3, 3]).on('zoom', (event) => {
    gRoot.attr('transform', event.transform);
});
svg.call(zoom);

const centerBtn = document.getElementById('center-root');
centerBtn.addEventListener('click', () => {
    centerOnContent();
});

function sizeSVG() {
    const rect = svg.node().getBoundingClientRect();
    return { width: rect.width, height: rect.height };
}

function computePositions(root, { levelHeight = 140, nodeSpacing = 180, minSpacing = 160, verticalOffset = 40 } = {}) {
    const { width, height } = sizeSVG();
    const cx = width / 2;
    const startY = 80; // Start from top
    const margin = 50; // Side margins

    const nodes = [];
    const links = [];
    const levels = [];

    // Level 0: Root
    levels[0] = [{ depth: 0, x: cx, y: startY, data: { id: 'root', title: root.title || root.id } }];

    // Level 1: Direct children of root
    const level1 = root.children || [];
    if (level1.length > 0) {
        levels[1] = level1.map((n, i) => ({
            depth: 1,
            x: 0, // Will be positioned later
            y: startY + levelHeight,
            data: n,
            parent: levels[0][0]
        }));
    }

    // Level 2+: Children of level 1 nodes
    if (levels[1]) {
        levels[1].forEach((parentNode, parentIndex) => {
            const children = parentNode.data.children || [];
            if (children.length > 0) {
                const level2Start = startY + (levelHeight * 2);
                children.forEach((child, childIndex) => {
                    if (!levels[2]) levels[2] = [];
                    levels[2].push({
                        depth: 2,
                        x: 0, // Will be positioned later
                        y: level2Start,
                        data: child,
                        parent: parentNode
                    });
                });
            }
        });
    }

    // Position nodes with collision detection and flexible vertical positioning
    levels.forEach((level, levelIndex) => {
        if (!level || level.length === 0) return;

        const levelWidth = width - (margin * 2);
        const availableSpacing = Math.max(minSpacing, levelWidth / level.length);
        
        // Center the nodes horizontally initially
        const totalWidth = (level.length - 1) * availableSpacing;
        const startX = cx - (totalWidth / 2);

        // Position nodes with collision detection
        const positionedNodes = [];
        level.forEach((node, index) => {
            let x = startX + (index * availableSpacing);
            let y = node.y;
            let attempts = 0;
            const maxAttempts = 20;

            // Try to find a non-overlapping position
            while (attempts < maxAttempts) {
                let hasCollision = false;
                
                // Check collision with already positioned nodes
                for (const existing of positionedNodes) {
                    const distance = Math.sqrt(Math.pow(node.x - existing.x, 2) + Math.pow(node.y - existing.y, 2));
                    if (distance < minSpacing) {
                        hasCollision = true;
                        break;
                    }
                }

                if (!hasCollision) break;

                // Try different positions: alternate between horizontal and vertical adjustments
                if (attempts % 2 === 0) {
                    // Horizontal adjustment
                    const offset = Math.ceil(attempts / 4) * (attempts % 4 < 2 ? 1 : -1) * (minSpacing * 0.8);
                    x = Math.max(margin, Math.min(width - margin, x + offset));
                } else {
                    // Vertical adjustment within the level range
                    const baseY = node.y;
                    const offset = Math.ceil(attempts / 4) * (attempts % 4 < 2 ? 1 : -1) * verticalOffset;
                    y = Math.max(baseY - verticalOffset, Math.min(baseY + verticalOffset, y + offset));
                }

                node.x = x;
                node.y = y;
                attempts++;
            }

            // If still colliding, use a simple grid layout with vertical offset
            if (attempts >= maxAttempts) {
                const totalNodes = level.length;
                const spacing = Math.max(minSpacing, (width - 200) / totalNodes);
                x = 100 + (index * spacing);
                y = node.y + (index % 2 === 0 ? 0 : verticalOffset * 0.5);
                node.x = x;
                node.y = y;
            }

            positionedNodes.push(node);
            nodes.push(node);

            // Add link to parent
            if (node.parent) {
                links.push({
                    depth: node.depth,
                    source: { x: node.parent.x, y: node.parent.y },
                    target: { x: node.x, y: node.y }
                });
            }
        });
    });

    return { nodes, links };
}

function renderMindMap(data) {
    // Hide welcome screen and show mind map
    document.getElementById('welcome-screen').style.display = 'none';
    document.getElementById('mindmap').style.display = 'block';
    
    const hierarchy = toHierarchy(data);
    const { nodes, links } = computePositions(hierarchy);

    // Straight line connector path
    function linePath(sx, sy, tx, ty) {
        return `M ${sx},${sy} L ${tx},${ty}`;
    }

    const linkSel = gLinks.selectAll('path').data(links, d => `${d.source.x},${d.source.y}->${d.target.x},${d.target.y}`);
    linkSel.join(
        enter => enter.append('path')
            .attr('class', d => `link ${d.depth === 1 ? 'link--lvl1' : 'link--lvl2'}`)
            .attr('d', d => linePath(d.source.x, d.source.y, d.target.x, d.target.y))
            .attr('stroke-width', d => d.depth === 1 ? 1.6 : 1.2)
            .attr('stroke-linecap', 'round')
            .attr('opacity', 0)
            .transition().duration(550)
            .attr('opacity', 1),
        update => update
            .transition().duration(550)
            .attr('d', d => linePath(d.source.x, d.source.y, d.target.x, d.target.y)),
        exit => exit.transition().duration(250).attr('opacity', 0).remove()
    );

    const nodeSel = gNodes.selectAll('g.node').data(nodes, d => d.data.id || d.data.title);
    const nodeEnter = nodeSel.enter().append('g').attr('class', d => {
        const base = 'node';
        return base + ' ' + (d.depth === 0 ? 'node--root' : d.depth === 1 ? 'node--lvl1' : 'node--lvl2');
    });

    nodeEnter.attr('transform', d => `translate(${d.x},${d.y})`).attr('opacity', 1);

    // Draw rounded rectangles sized to text (defer measuring until layout is ready)
    const paddingX = 16;
    const paddingY = 10;
    const corner = 12;

    nodeEnter.each(function(d) {
        const g = d3.select(this);
        // insert rect with provisional size
        const rect = g.insert('rect', ':first-child')
            .attr('x', -40)
            .attr('y', -18)
            .attr('rx', corner)
            .attr('ry', corner)
            .attr('width', 80)
            .attr('height', 36);

        const title = g.append('text')
            .attr('class', 'node__label')
            .attr('text-anchor', 'middle')
            .text(d.data.title);

        // add badge placeholder
        const badge = (d.depth === 0 || d.data.difficulty)
            ? g.append('text').attr('class', 'node__badge').attr('text-anchor', 'middle').text(d.depth === 0 ? 'ROOT' : (d.data.difficulty || ''))
            : null;

        // measure after next frame to ensure text is laid out
        requestAnimationFrame(() => {
            const bbox = title.node().getBBox();
            const width = Math.max(64, bbox.width + paddingX * 2);
            const height = Math.max(32, bbox.height + paddingY * 2 + (badge ? 12 : 0));
            rect
                .attr('x', -width / 2)
                .attr('y', -height / 2)
                .attr('width', width)
                .attr('height', height);
            title.attr('y', badge ? -4 : 0);
            if (badge) {
                badge.attr('y', (height / 2) - 8);
            }
        });
    });

    const nodeAll = nodeEnter.merge(nodeSel);
    nodeAll.attr('transform', d => `translate(${d.x},${d.y})`);

    // Update texts and resize rects on update
    nodeAll.each(function(d) {
        const g = d3.select(this);
        let title = g.select('text.node__label');
        if (title.empty()) {
            title = g.append('text').attr('class', 'node__label').attr('text-anchor', 'middle');
        }
        title.text(d.data.title);

        let rect = g.select('rect');
        if (rect.empty()) {
            rect = g.insert('rect', ':first-child');
        }

        let badge = g.select('text.node__badge');
        if (d.data.difficulty) {
            if (badge.empty()) badge = g.append('text').attr('class', 'node__badge').attr('text-anchor', 'middle');
            badge.text(d.data.difficulty);
        } else if (!badge.empty()) {
            badge.remove();
            badge = g.select('text.node__badge');
        }

        // measure after text set
        const bbox = title.node().getBBox();
        const paddingX = 16; const paddingY = 10; const corner = 12;
        const width = Math.max(64, bbox.width + paddingX * 2);
        const height = Math.max(32, bbox.height + paddingY * 2 + (!badge.empty() ? 12 : 0));
        rect
            .attr('x', -width / 2)
            .attr('y', -height / 2)
            .attr('rx', corner)
            .attr('ry', corner)
            .attr('width', width)
            .attr('height', height);
        title.attr('y', !badge.empty() ? -4 : 0);
        if (!badge.empty()) badge.attr('y', (height / 2) - 8);
    });

    nodeSel.exit().transition().duration(250).attr('opacity', 0).remove();

    // interactions
    enableInteractions(nodeAll);

    // After render, center the entire content in view
    centerOnContent();
}

// -------------------------------
// Interactions
// -------------------------------
const tooltip = createTooltip();
function createTooltip() {
    const el = document.createElement('div');
    el.className = 'tooltip';
    el.style.display = 'none';
    document.body.appendChild(el);
    return el;
}

function enableInteractions(sel) {
    sel.on('mouseenter', function (event, d) {
        // Only update sidebar details, no tooltip on main screen
        const details = document.getElementById('node-details');
        details.innerHTML = sidebarHTML(d.data);
    }).on('click', function (event, d) {
        // focus zoom to node
        const t = d3.zoomTransform(svg.node());
        const scale = Math.min(2.2, Math.max(0.6, t.k));
        const { width, height } = sizeSVG();
        const tx = width / 2 - d.x * scale;
        const ty = height / 2 - d.y * scale;
        const transform = d3.zoomIdentity.translate(tx, ty).scale(scale);
        svg.transition().duration(650).call(zoom.transform, transform);
    });
}

function positionTooltip(event) {
    tooltip.style.left = `${event.clientX}px`;
    tooltip.style.top = `${event.clientY}px`;
}

function tooltipHTML(node) {
    const meta = [];
    if (node.difficulty) meta.push(`Difficulty: ${node.difficulty}`);
    if (node.time_left) meta.push(`Time: ${node.time_left}`);
    return `
        <div class="title">${escapeHTML(node.title || 'Root')}</div>
        <div class="meta">${escapeHTML(meta.join(' • ') || 'Node')}</div>
        <div>${escapeHTML(node.description || '')}</div>
    `;
}

function sidebarHTML(node) {
    return `
        <div><strong>${escapeHTML(node.title || 'Root')}</strong></div>
        ${node.description ? `<div class="muted" style="margin-top:6px;">${escapeHTML(node.description)}</div>` : ''}
        <div style="margin-top:8px; font-size:12px;">
            ${node.difficulty ? `<span>Difficulty: <strong>${escapeHTML(node.difficulty)}</strong></span>` : ''}
            ${node.time_left ? `<span style="margin-left:10px;">Time: <strong>${escapeHTML(node.time_left)}</strong></span>` : ''}
        </div>
    `;
}

function escapeHTML(str) {
    return String(str).replace(/[&<>"']/g, (m) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[m]));
}

// -------------------------------
// Form wiring
// -------------------------------
const form = document.getElementById('prompt-form');
const input = document.getElementById('prompt-input');
const submitBtn = form.querySelector('button[type="submit"]');
let requestSeq = 0;

async function updateMindMapFromInput(raw) {
    const seq = ++requestSeq;
    try {
        submitBtn.disabled = true;
        submitBtn.textContent = 'Generating...';
        let data;
        if (raw && raw.trim().startsWith('{')) {
            data = JSON.parse(raw);
        } else {
            data = await fetchMindMap(raw || '');
        }
        if (seq !== requestSeq) return; // stale
        renderMindMap(data);
    } catch (err) {
        console.error(err);
        alert('Failed to generate mind map.');
    } finally {
        if (seq === requestSeq) {
            submitBtn.disabled = false;
            submitBtn.textContent = 'Generate';
        }
    }
}

// No debounce; refresh only on Generate click
form.addEventListener('submit', async (e) => {
    e.preventDefault();
    updateMindMapFromInput(input.value);
});

// Welcome screen typing animation
const pitchSentences = [
    "Transform your ideas into structured learning paths...",
    "AI-powered mind maps for complex topics...",
    "Break down any subject into digestible steps...",
    "From concept to mastery with intelligent organization...",
    "Turn overwhelming topics into clear roadmaps...",
    "Your personal learning companion powered by AI..."
];

let currentSentenceIndex = 0;
let currentCharIndex = 0;
let isDeleting = false;
let typingSpeed = 100;

function typeWriter() {
    const typingContent = document.getElementById('typing-content');
    const currentSentence = pitchSentences[currentSentenceIndex];
    
    if (isDeleting) {
        typingContent.textContent = currentSentence.substring(0, currentCharIndex - 1);
        currentCharIndex--;
        typingSpeed = 50;
    } else {
        typingContent.textContent = currentSentence.substring(0, currentCharIndex + 1);
        currentCharIndex++;
        typingSpeed = 100;
    }
    
    if (!isDeleting && currentCharIndex === currentSentence.length) {
        typingSpeed = 2000; // Pause at end
        isDeleting = true;
    } else if (isDeleting && currentCharIndex === 0) {
        isDeleting = false;
        currentSentenceIndex = (currentSentenceIndex + 1) % pitchSentences.length;
        typingSpeed = 500; // Pause before next sentence
    }
    
    setTimeout(typeWriter, typingSpeed);
}

// Start typing animation
typeWriter();

// Auto-fit view to rendered content
const observer = new ResizeObserver(() => {
    // Recenter on resize to keep content visible
    svg.call(zoom.transform, d3.zoomIdentity);
});
observer.observe(document.getElementById('mindmap'));

function centerOnRoot(nodes, scale = 1) {
    const { width, height } = sizeSVG();
    const rootNode = nodes.find(n => n.depth === 0);
    const tx = width / 2 - (rootNode ? rootNode.x : width / 2) * scale;
    const ty = height / 2 - (rootNode ? rootNode.y : height / 2) * scale;
    const transform = d3.zoomIdentity.translate(tx, ty).scale(scale);
    svg.transition().duration(550).call(zoom.transform, transform);
}

function centerOnContent() {
    try {
        const { width, height } = sizeSVG();
        const bbox = gRoot.node().getBBox();
        
        // Add padding around the content
        const padding = 60;
        const availableWidth = width - (padding * 2);
        const availableHeight = height - (padding * 2);
        
        // Calculate scale to fit content in frame
        const scaleX = availableWidth / bbox.width;
        const scaleY = availableHeight / bbox.height;
        const scale = Math.min(scaleX, scaleY, 3); // Max scale of 3 to prevent too large
        
        // Center the content
        const cx = bbox.x + bbox.width / 2;
        const cy = bbox.y + bbox.height / 2;
        const tx = width / 2 - cx * scale;
        const ty = height / 2 - cy * scale;
        
        const transform = d3.zoomIdentity.translate(tx, ty).scale(scale);
        svg.transition().duration(550).call(zoom.transform, transform);
    } catch (e) {
        // ignore when no content yet
    }
}

