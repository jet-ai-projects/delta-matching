"use strict";

(() => {
  const root = document.documentElement;
  const themeButtons = [...document.querySelectorAll(".theme-toggle[data-theme-choice]")];
  const syncThemeButtons = () => {
    themeButtons.forEach((button) => {
      const selected = button.dataset.themeChoice === root.dataset.theme;
      button.setAttribute("aria-pressed", String(selected));
    });
  };
  document.addEventListener("delta-theme-change", syncThemeButtons);
  themeButtons.forEach((button) => {
    button.addEventListener("click", () => {
      if (window.deltaMatchingTheme) window.deltaMatchingTheme.set(button.dataset.themeChoice);
      else root.dataset.theme = button.dataset.themeChoice;
      syncThemeButtons();
    });
  });
  syncThemeButtons();

  const dialog = document.querySelector(".figure-dialog");
  const dialogImage = dialog?.querySelector("img");
  const dialogCaption = dialog?.querySelector(".dialog-caption");
  let figureOpener = null;

  if (dialog && dialogImage && dialogCaption && typeof dialog.showModal === "function") {
    document.querySelectorAll(".figure-link").forEach((link) => {
      link.addEventListener("click", (event) => {
        if (event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
        const image = link.querySelector("img");
        if (!image) return;
        event.preventDefault();
        figureOpener = link;
        dialogImage.src = link.href;
        dialogImage.alt = image.alt;
        dialogCaption.textContent = link.closest("figure")?.querySelector("figcaption")?.textContent || image.alt;
        dialog.showModal();
        document.body.classList.add("figure-open");
        dialog.querySelector(".dialog-close")?.focus();
      });
    });
    dialog.querySelector(".dialog-close")?.addEventListener("click", () => dialog.close());
    dialog.addEventListener("click", (event) => {
      if (event.target !== dialog) return;
      const bounds = dialog.getBoundingClientRect();
      const outside = event.clientX < bounds.left || event.clientX > bounds.right || event.clientY < bounds.top || event.clientY > bounds.bottom;
      if (outside) dialog.close();
    });
    dialog.addEventListener("close", () => {
      document.body.classList.remove("figure-open");
      figureOpener?.focus({ preventScroll: true });
      figureOpener = null;
    });
  }

  const progress = document.querySelector(".reading-progress");
  const sections = [...document.querySelectorAll(".article-section")];
  const tocLinks = [...document.querySelectorAll(".toc-nav a")];
  let framePending = false;

  const setActiveSection = (id) => {
    tocLinks.forEach((link) => {
      const active = link.hash === `#${id}`;
      link.classList.toggle("is-active", active);
      if (active) link.setAttribute("aria-current", "location");
      else link.removeAttribute("aria-current");
    });
  };

  const updateProgress = () => {
    framePending = false;
    const scrollable = document.documentElement.scrollHeight - window.innerHeight;
    const ratio = scrollable > 0 ? Math.min(1, Math.max(0, window.scrollY / scrollable)) : 0;
    if (progress) progress.style.width = `${ratio * 100}%`;
    // Use the native anchor offset, with a small tolerance for fractional pixels.
    const threshold = (parseFloat(getComputedStyle(root).scrollPaddingTop) || 0) + 2;
    let active = sections[0];
    for (const section of sections) {
      if (section.getBoundingClientRect().top <= threshold) active = section;
      else break;
    }
    if (active) setActiveSection(active.id);
  };

  const requestProgress = () => {
    if (framePending) return;
    framePending = true;
    window.requestAnimationFrame(updateProgress);
  };
  window.addEventListener("scroll", requestProgress, { passive: true });
  window.addEventListener("resize", requestProgress);
  window.addEventListener("load", requestProgress);
  window.addEventListener("hashchange", requestProgress);

  if ("IntersectionObserver" in window) {
    const observer = new IntersectionObserver(requestProgress, { rootMargin: "-15% 0px -65% 0px", threshold: 0 });
    sections.forEach((section) => observer.observe(section));
  }
  document.querySelectorAll(".mobile-toc a").forEach((link) => {
    link.addEventListener("click", () => {
      const details = link.closest("details");
      if (details) details.open = false;
    });
  });
  updateProgress();
})();
