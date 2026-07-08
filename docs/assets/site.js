const current = location.pathname.split("/").pop() || "index.html";
document.querySelectorAll(".nav-links a").forEach((link) => {
  const href = link.getAttribute("href");
  if (href === current || (current === "" && href === "index.html")) {
    link.classList.add("active");
  }
});

function text(value) {
  if (value === null || value === undefined || value === "") return "-";
  if (Array.isArray(value)) return value.join(", ");
  return String(value);
}

async function renderParameters() {
  const mount = document.querySelector("[data-parameters]");
  if (!mount) return;
  const response = await fetch("assets/parameters.json");
  const parameters = await response.json();
  const search = document.querySelector("[data-param-search]");
  const mode = document.querySelector("[data-param-mode]");
  const section = document.querySelector("[data-param-section]");
  const sections = [...new Set(parameters.map((p) => p.section_id || p.group).filter(Boolean))].sort();
  sections.forEach((name) => {
    const option = document.createElement("option");
    option.value = name;
    option.textContent = name;
    section.appendChild(option);
  });

  function draw() {
    const q = (search.value || "").toLowerCase();
    const modeValue = mode.value;
    const sectionValue = section.value;
    const rows = parameters.filter((p) => {
      const blob = [p.name, p.label, p.cli_tag, p.description, p.section_id, p.group].join(" ").toLowerCase();
      return (!q || blob.includes(q))
        && (!modeValue || p.mode === modeValue)
        && (!sectionValue || p.section_id === sectionValue || p.group === sectionValue);
    });
    mount.innerHTML = `
      <table class="param-table">
        <thead>
          <tr>
            <th>Name</th><th>CLI</th><th>Default</th><th>Type</th><th>Mode</th><th>Description</th>
          </tr>
        </thead>
        <tbody>
          ${rows.map((p) => `
            <tr>
              <td><span class="param-name">${text(p.label)}</span><br><span class="muted">${text(p.name)}</span></td>
              <td><code>${text(p.cli_tag)}</code></td>
              <td><code>${text(p.default)}</code></td>
              <td>${text(p.type)}${p.options && p.options.length ? `<br><span class="muted">${text(p.options)}</span>` : ""}</td>
              <td>${text(p.mode)}<br><span class="muted">${text(p.section_id || p.group)}</span></td>
              <td>${text(p.description)}</td>
            </tr>
          `).join("")}
        </tbody>
      </table>
    `;
  }

  [search, mode, section].forEach((el) => el.addEventListener("input", draw));
  draw();
}

renderParameters().catch((err) => {
  const mount = document.querySelector("[data-parameters]");
  if (mount) mount.innerHTML = `<p class="callout">Could not load parameter manual: ${err}</p>`;
});
