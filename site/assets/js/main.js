(() => {
  // Highlight code only when a block is first opened (keeps page load light).
  let hljsReady;
  const loadHljs = () =>
    (hljsReady ??= Promise.all([
      import("./vendor/hljs-core.min.js"),
      import("./vendor/hljs-python.min.js"),
    ]).then(([core, py]) => {
      core.default.registerLanguage("python", py.default);
      return core.default;
    }));
  document.querySelectorAll("details.code-block").forEach((d) => {
    d.addEventListener("toggle", async () => {
      if (!d.open || d.dataset.hl) return;
      d.dataset.hl = "1";
      const el = d.querySelector("pre code");
      try {
        (await loadHljs()).highlightElement(el);
      } catch {
        /* plain code is still readable */
      }
    });
  });

  // Copy buttons
  document.querySelectorAll(".code-wrap .copy").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const text = btn.parentElement.querySelector("code").innerText;
      try {
        await navigator.clipboard.writeText(text);
        btn.textContent = "Copied";
      } catch {
        btn.textContent = "Copy failed";
      }
      setTimeout(() => (btn.textContent = "Copy"), 1600);
    });
  });

  // Lightbox for figures
  const lb = document.getElementById("lightbox");
  const lbImg = lb.querySelector("img");
  document.querySelectorAll(".fig .zoom").forEach((b) => {
    b.addEventListener("click", () => {
      const img = b.querySelector("img");
      lbImg.src = img.currentSrc || img.src;
      lbImg.alt = img.alt;
      lb.showModal();
    });
  });
  lb.addEventListener("click", () => lb.close());

  // Only one video plays at a time
  const vids = [...document.querySelectorAll("video[controls]")];
  vids.forEach((v) => v.addEventListener("play", () => vids.forEach((o) => o !== v && o.pause())));

  // Mobile contents menu
  const btn = document.querySelector(".menu-btn");
  const setMenu = (open) => {
    document.body.classList.toggle("menu-open", open);
    btn.setAttribute("aria-expanded", String(open));
  };
  btn.addEventListener("click", (e) => {
    e.stopPropagation();
    setMenu(!document.body.classList.contains("menu-open"));
  });
  document.querySelectorAll(".sidebar a").forEach((a) => a.addEventListener("click", () => setMenu(false)));
  document.addEventListener("click", (e) => {
    if (document.body.classList.contains("menu-open") && !e.target.closest(".sidebar")) setMenu(false);
  });
  document.addEventListener("keydown", (e) => e.key === "Escape" && setMenu(false));

  // Mark the current section in the contents list
  const links = new Map([...document.querySelectorAll(".toc a")].map((a) => [a.hash.slice(1), a]));
  const secs = [...links.keys()].map((id) => document.getElementById(id)).filter(Boolean);
  const mark = () => {
    let cur = secs[0];
    for (const s of secs) if (s.getBoundingClientRect().top < window.innerHeight * 0.35) cur = s;
    links.forEach((a, id) => a.setAttribute("aria-current", String(id === cur.id)));
  };
  addEventListener("scroll", mark, { passive: true });
  mark();
})();
