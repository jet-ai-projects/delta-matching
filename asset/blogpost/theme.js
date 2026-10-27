"use strict";

// Apply the saved theme, or light by default, before stylesheets are painted.
(() => {
  const root = document.documentElement;
  const storageKey = "delta-matching-theme";
  const validTheme = (value) => value === "light" || value === "dark";
  let preference = null;

  try {
    const saved = window.localStorage.getItem(storageKey);
    if (validTheme(saved)) preference = saved;
  } catch (_) {
    // Theme switching also works when browser storage is unavailable.
  }

  const apply = () => {
    const theme = preference ?? "light";
    root.dataset.theme = theme;
    root.style.colorScheme = theme;
    document.dispatchEvent(new CustomEvent("delta-theme-change", { detail: { theme } }));
  };

  window.deltaMatchingTheme = {
    set(theme) {
      if (!validTheme(theme)) return;
      preference = theme;
      try {
        window.localStorage.setItem(storageKey, theme);
      } catch (_) {
        // Retain the selected theme for this page even without persistence.
      }
      apply();
    },
  };

  window.addEventListener("storage", (event) => {
    if (event.key !== storageKey && event.key !== null) return;
    preference = validTheme(event.newValue) ? event.newValue : null;
    apply();
  });
  apply();
})();
