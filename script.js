const projects = [
  {
    id: "robot-composer",
    title: "AI-Powered Robot Composer",
    year: "2024-2025",
    tags: ["AI", "Python", "Evaluation"],
    stack: ["Python", "PyTorch", "Music21", "LSTM", "VAE", "MIDI"],
    summary: "Final-year emotion-conditioned MIDI generator with short-term and long-term latent modelling.",
    evidence: [
      "Designed a 12.55-million-parameter Hierarchical VAE with recurrent short-term and phrase-level representations conditioned on valence, arousal, and dominance.",
      "Trained over 310,748 sequence windows using a 4,343-token note/chord vocabulary and 11 duration classes.",
      "Packaged the original checkpoint into a tested CLI with reproducible MIDI generation, eight representative VAD samples, and a results report; the original notebook recorded 1.7574 note perplexity."
    ],
    lesson: "Generative systems need more than an interesting architecture. The conditioning signal, artefact pipeline, objective metrics, and human evaluation all need to remain inspectable after the experiment ends.",
    codeLink: "code-evidence/ai-robot-composer/README.md",
    visual: [46, 58, 70, 84, 62, 92, 76, 88, 68, 96]
  },
  {
    id: "ecommerce",
    title: "Full-Stack E-Commerce Platform",
    year: "2023",
    tags: ["Full-stack", "Backend", "Team"],
    stack: ["Flask", "SQL", "JavaScript", "HTML/CSS", "Git"],
    summary: "Team coursework product with authentication, product listing, and payment-related flows.",
    evidence: [
      "Led back-end development and translated feature requirements into database-backed application logic.",
      "Integrated Flask and SQL workflows with JavaScript, HTML, and CSS front-end components.",
      "Worked with four teammates using Git, Agile planning, testing, and CI/CD practices."
    ],
    lesson: "The useful part of full-stack work is not only writing endpoints; it is making a user flow hold together across data, interface, and team coordination.",
    visual: [34, 58, 72, 44, 86, 52, 68, 40, 78, 60]
  },
  {
    id: "rag",
    title: "Retrieval-Augmented Generation QA Pipeline",
    year: "2026",
    tags: ["AI", "Data", "Evaluation"],
    stack: ["Python", "sentence-transformers", "BM25", "Reranking", "JSON"],
    summary: "Multi-stage RAG pipeline with structured outputs and full component comparison.",
    evidence: [
      "Built corpus preparation, chunking, embedding generation, retrieval, reranking, prompt variants, and answer evaluation.",
      "Evaluated 84 valid configurations across 90 gold-answer question-answer pairs.",
      "Best configuration achieved 0.342 ROUGE-L, about 30 percent above a 0.263 baseline."
    ],
    lesson: "AI features need measurable baselines, inspectable outputs, and iteration loops. A model call is only one part of the product.",
    visual: [42, 50, 66, 74, 62, 88, 55, 70, 80, 92]
  },
  {
    id: "java-testing",
    title: "Java Game Testing and Behaviour Specification",
    year: "2025",
    tags: ["Testing", "Java", "Debugging"],
    stack: ["Java 17", "Maven", "Cucumber", "JUnit", "BDD"],
    summary: "Existing Java codebase extended with behaviour-driven feature tests and generated reports.",
    evidence: [
      "Added Cucumber feature specifications and a JUnit runner in a Maven Java 17 project.",
      "Covered behaviours including snake direction changes, personal high scores, and unlockable tree toggling.",
      "Turned expected behaviour into repeatable checks while reading unfamiliar code."
    ],
    lesson: "Good debugging starts by making the expected behaviour precise enough to test, then tightening the loop until the failure is explainable.",
    visual: [28, 36, 54, 48, 76, 62, 90, 74, 60, 82]
  },
  {
    id: "cpp-library",
    title: "C++ Library Management System",
    year: "2024",
    tags: ["Systems", "Testing", "C++"],
    stack: ["C++17", "Make", "Catch", "Static analysis", "CSV"],
    summary: "Object-oriented C++ system with borrowing logic, CSV export, and automated tests.",
    evidence: [
      "Implemented document subclasses, dynamic dispatch, copy and move semantics, and borrowing/returning logic.",
      "Validated the system with a Makefile, Catch tests, compiler warnings, static analysis, and clang-tidy targets.",
      "Kept implementation behaviour visible through CSV export and repeatable test runs."
    ],
    lesson: "Lower-level software makes design trade-offs harder to hide. Ownership, copying, and validation need to be deliberate.",
    visual: [52, 46, 80, 68, 42, 74, 58, 88, 64, 76]
  },
  {
    id: "fashion-mnist",
    title: "Efficient Fashion-MNIST Classifier",
    year: "2025",
    tags: ["AI", "Python", "Docker"],
    stack: ["Python", "PyTorch", "Docker", "CNNs", "Validation"],
    summary: "Compact image classifier with reproducible packaging under a parameter constraint.",
    evidence: [
      "Implemented a PyTorch CNN and training loop with validation tracking and saved weights.",
      "Used Docker-based submission requirements and scripts to verify loading, prediction format, and training behaviour.",
      "Best recorded validation accuracy reached 91.8 percent under a 100k-parameter constraint."
    ],
    lesson: "Constraints are useful. They push the work away from brute force and towards model design, measurement, and reproducibility.",
    visual: [38, 44, 57, 63, 70, 82, 88, 91, 76, 84]
  }
];

const filters = ["All", "Full-stack", "AI", "Testing", "Systems", "Backend", "Data"];
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
  if (activeFilter === "All") {
    return projects;
  }

  return projects.filter((project) => project.tags.includes(activeFilter));
}

function renderProjectCards() {
  const visibleProjects = getVisibleProjects();

  projectGrid.innerHTML = visibleProjects
    .map((project) => {
      const selectedClass = project.id === selectedProjectId ? " is-selected" : "";
      const bars = project.visual
        .map((height) => `<span style="height:${height}%"></span>`)
        .join("");
      const stack = project.stack
        .map((item) => `<span class="stack-pill">${item}</span>`)
        .join("");

      return `
        <article class="project-card${selectedClass}" tabindex="0" data-project-id="${project.id}">
          <div class="project-visual" aria-hidden="true">${bars}</div>
          <div class="project-meta">
            <span>${project.year}</span>
            <span>${project.tags.join(" / ")}</span>
          </div>
          <h2>${project.title}</h2>
          <p>${project.summary}</p>
          <div class="stack-list">${stack}</div>
        </article>
      `;
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
  const codeLink = project.codeLink
    ? `<a class="detail-action" href="${project.codeLink}">View code evidence</a>`
    : "";

  detailPanel.innerHTML = `
    <p class="detail-kicker">${project.tags.join(" / ")}</p>
    <h2>${project.title}</h2>
    <p>${project.summary}</p>
    <ul class="detail-list">${evidence}</ul>
    <div class="lesson">
      <strong>What I learned</strong>
      ${project.lesson}
    </div>
    ${codeLink}
  `;
}

function render() {
  renderFilters();
  renderProjectCards();
  renderDetailPanel();
}

render();
