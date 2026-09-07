const projects = [
  {
    id: "music-recommender",
    title: "MSc Music Recommender",
    year: "2026",
    tags: ["AI", "Data", "Evaluation", "Python"],
    stack: ["Python", "PyTorch", "LightGCN", "Ranking", "Fairness"],
    summary: "Graph-based music recommendation with popularity-aware reranking.",
    evidence: [
      "Prepared 751,531 implicit interactions from 5,199 users and 8,640 items for graph collaborative filtering.",
      "Compared LightGCN depth and embedding size using held-out ranking evaluation and repeated seeds.",
      "The selected one-layer, 128-dimensional model reached mean test NDCG@20 of 0.1294; Smooth xQuAD increased catalogue coverage while keeping the ranking loss small."
    ],
    lesson: "A recommendation model needs an evaluation protocol that makes the ranking task and the trade-offs visible. Accuracy and catalogue exposure need to be read together.",
    codeLink: "code-evidence/msc-music-recommender/README.md",
    visual: [38, 54, 66, 48, 80, 62, 74, 90, 68, 84]
  },
  {
    id: "hkl",
    title: "HKL Racing Prediction",
    year: "2026",
    tags: ["AI", "Data", "Evaluation", "Python"],
    stack: ["Python", "scikit-learn", "Feature engineering", "Testing"],
    summary: "Personal tabular machine-learning pipeline for top-four race prediction.",
    evidence: [
      "Built a time-aware validation pipeline over 107,033 historical race rows.",
      "Added race-level rank, field-size, and implied-probability features before training several classifiers.",
      "The Random Forest reached 69.4% balanced accuracy against a 50.0% balanced-accuracy baseline."
    ],
    lesson: "A useful model result depends on the split strategy, the baseline, and the limits of what the data can support.",
    codeLink: "code-evidence/HKL_Prediction_Project/README.md",
    visual: [48, 62, 56, 80, 72, 88, 64, 76, 92, 68]
  },
  {
    id: "robot-composer",
    title: "AIVAD Robot Composer",
    year: "2024-2025",
    tags: ["AI", "Evaluation", "Python"],
    stack: ["Python", "PyTorch", "LSTM", "VAE", "MIDI"],
    summary: "Emotion-conditioned piano generation using valence, arousal, and dominance.",
    evidence: [
      "Designed a 12.55-million-parameter hierarchical VAE with short-term and long-term latent representations.",
      "Packaged the trained checkpoint into a tested CLI with MIDI generation, analysis, and representative samples.",
      "The original evaluation recorded note perplexity of 1.7574 and the curated repository independently checks the included MIDI outputs."
    ],
    lesson: "Generative systems are easier to understand when the model, output artefacts, and evaluation measures remain inspectable after training.",
    codeLink: "code-evidence/ai-robot-composer/README.md",
    visual: [46, 58, 70, 84, 62, 92, 76, 88, 68, 96]
  },
  {
    id: "rag",
    title: "Retrieval-Augmented Generation Pipeline",
    year: "2026",
    tags: ["AI", "Data", "Evaluation", "Python"],
    stack: ["Python", "Embeddings", "BM25", "Reranking", "Transformers"],
    summary: "A multi-stage question-answering pipeline evaluated across retrieval and prompt choices.",
    evidence: [
      "Compared section-aware and sliding-window chunking, dense embeddings, BM25, cross-encoder reranking, and four prompt strategies.",
      "Evaluated 84 configurations over 90 benchmark questions.",
      "The best recorded configuration achieved ROUGE-L 0.342 compared with 0.263 for the baseline."
    ],
    lesson: "A language model is only one part of a useful question-answering system. Retrieval quality, prompt design, and measurement all affect the result.",
    codeLink: "code-evidence/rag-pipeline/README.md",
    visual: [42, 50, 66, 74, 62, 88, 55, 70, 80, 92]
  },
  {
    id: "data-engineering",
    title: "Data Engineering Coursework",
    year: "2025-2026",
    tags: ["Data", "Python", "Testing"],
    stack: ["SQLite", "MongoDB", "Pandas", "Snakemake", "CSV/JSON"],
    summary: "Four practical labs covering data cleaning, integration, analysis, and workflow automation.",
    evidence: [
      "Worked with SQLite databases and transport data, including cleaning and query tasks.",
      "Integrated CSV, JSON, MongoDB, and SQLite data sources in a journey-time analysis.",
      "Built a Snakemake workflow with configuration, validation, notebooks, and environment definitions."
    ],
    lesson: "Data work becomes more reliable when the transformations, inputs, and execution order are explicit.",
    codeLink: "code-evidence/comp63301-data-engineering-concepts/README.md",
    visual: [32, 48, 70, 58, 82, 66, 90, 74, 54, 78]
  },
  {
    id: "fashion-mnist",
    title: "Compact Fashion-MNIST CNN",
    year: "2025",
    tags: ["AI", "Evaluation", "Python"],
    stack: ["Python", "PyTorch", "CNN", "Validation"],
    summary: "A compact image classifier with augmentation and repeated-split tuning.",
    evidence: [
      "Implemented three convolutional blocks with batch normalisation, pooling, global average pooling, and a small classifier head.",
      "Compared learning-rate and weight-decay settings over repeated train-validation splits.",
      "Kept the training loop, validation history, and weight export separate so the model can be checked independently."
    ],
    lesson: "A parameter constraint made the model design and validation choices matter more than simply adding capacity.",
    codeLink: "code-evidence/fashion-mnist/README.md",
    visual: [38, 44, 57, 63, 70, 82, 88, 91, 76, 84]
  },
  {
    id: "dbpedia",
    title: "DBpedia Column Annotation",
    year: "2025",
    tags: ["Data", "Evaluation", "Python"],
    stack: ["Python", "Requests", "DBpedia", "SPARQL"],
    summary: "Entity lookup and ontology-aware class selection for table columns.",
    evidence: [
      "Sampled distinct column values and resolved their top DBpedia entities through the Lookup API.",
      "Compared majority-class selection with ontology expansion and coverage-specificity selection.",
      "Added handling for empty lookup results so a failed external query produces a visible result instead of a crash."
    ],
    lesson: "External knowledge-base systems need clear fallback behaviour because a network response can be empty even when the code is correct.",
    codeLink: "code-evidence/dbpedia-annotation/README.md",
    visual: [34, 52, 64, 78, 46, 70, 86, 60, 74, 90]
  }
];

const filters = ["All", "AI", "Data", "Evaluation", "Python", "Testing"];
let activeFilter = "All";
let selectedProjectId = projects[0].id;

const filterRow = document.querySelector("#filterRow");
const projectGrid = document.querySelector("#projectGrid");
const detailPanel = document.querySelector("#detailPanel");

function renderFilters() {
  filterRow.innerHTML = filters
    .map((filter) => {
      const activeClass = filter === activeFilter ? " is-active" : "";
      return `<button class="filter-button${activeClass}" type="button" data-filter="${filter}">${filter}</button>`;
    })
    .join("");

  filterRow.querySelectorAll("button").forEach((button) => {
    button.addEventListener("click", () => {
      activeFilter = button.dataset.filter;
      const visibleProjects = getVisibleProjects();
      selectedProjectId = visibleProjects[0]?.id || projects[0].id;
      render();
    });
  });
}

function getVisibleProjects() {
  return activeFilter === "All"
    ? projects
    : projects.filter((project) => project.tags.includes(activeFilter));
}

function renderProjectCards() {
  const visibleProjects = getVisibleProjects();
  projectGrid.innerHTML = visibleProjects
    .map((project) => {
      const selectedClass = project.id === selectedProjectId ? " is-selected" : "";
      const bars = project.visual.map((height) => `<span style="height:${height}%"></span>`).join("");
      const stack = project.stack.map((item) => `<span class="stack-pill">${item}</span>`).join("");
      return `
        <article class="project-card${selectedClass}" tabindex="0" data-project-id="${project.id}">
          <div class="project-visual" aria-hidden="true">${bars}</div>
          <div class="project-meta"><span>${project.year}</span><span>${project.tags.join(" / ")}</span></div>
          <h2>${project.title}</h2>
          <p>${project.summary}</p>
          <div class="stack-list">${stack}</div>
        </article>`;
    })
    .join("");

  projectGrid.querySelectorAll(".project-card").forEach((card) => {
    const selectCard = () => {
      selectedProjectId = card.dataset.projectId;
      renderProjectCards();
      renderDetailPanel();
    };
    card.addEventListener("click", selectCard);
    card.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        selectCard();
      }
    });
  });
}

function renderDetailPanel() {
  const project = projects.find((item) => item.id === selectedProjectId) || projects[0];
  const evidence = project.evidence.map((item) => `<li>${item}</li>`).join("");
  detailPanel.innerHTML = `
    <p class="detail-kicker">${project.tags.join(" / ")}</p>
    <h2>${project.title}</h2>
    <p>${project.summary}</p>
    <ul class="detail-list">${evidence}</ul>
    <div class="lesson"><strong>What I learned</strong>${project.lesson}</div>
    <a class="detail-action" href="${project.codeLink}">View code evidence</a>`;
}

function render() {
  renderFilters();
  renderProjectCards();
  renderDetailPanel();
}

render();
