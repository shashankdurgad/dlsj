// Team credits: the only place names live. They render in the hero and the footer.
// mark: "*" = University of Cambridge, "†" = UCL (see AFFILIATIONS).
const TEAM = [
  { name: "Desmond Zee", mark: "*" },
  { name: "Lucas Lim", mark: "†" },
  { name: "Shashank Durgad", mark: "†" },
  { name: "Wee Joe Tan", mark: "†" },
];
const AFFILIATIONS = [
  { mark: "*", name: "University of Cambridge" },
  { mark: "†", name: "UCL" },
];

const sup = (t) => Object.assign(document.createElement("sup"), { textContent: t });
for (const box of document.querySelectorAll("[data-team]")) {
  const names = document.createElement("ul");
  names.className = "team-names";
  for (const m of TEAM) {
    const li = document.createElement("li");
    li.append(m.name, sup(m.mark));
    names.append(li);
  }
  const affs = document.createElement("ul");
  affs.className = "team-affs";
  for (const a of AFFILIATIONS) {
    const li = document.createElement("li");
    li.append(sup(a.mark), a.name);
    affs.append(li);
  }
  box.append(names, affs);
}

// Section nav: mark the pill of the section in view.
const pills = new Map(
  [...document.querySelectorAll(".pills a")].map((a) => [a.getAttribute("href").slice(1), a])
);
const spy = new IntersectionObserver(
  (entries) => {
    for (const e of entries) {
      if (!e.isIntersecting) continue;
      for (const [id, a] of pills) {
        const on = id === e.target.id;
        a.classList.toggle("on", on);
        if (on) {
          a.setAttribute("aria-current", "true");
          const row = a.parentElement;
          if (row.scrollWidth > row.clientWidth)
            row.scrollTo({ left: a.offsetLeft - row.clientWidth / 2 + a.offsetWidth / 2 });
        } else a.removeAttribute("aria-current");
      }
    }
  },
  { rootMargin: "-45% 0px -50% 0px" }
);
for (const id of pills.keys()) {
  const s = document.getElementById(id);
  if (s) spy.observe(s);
}

// Looping clips play only while on screen, and never on their own under reduced motion.
const still = matchMedia("(prefers-reduced-motion: reduce)");
const clips = document.querySelectorAll("video[data-loop]");
if (still.matches) {
  for (const v of clips) {
    v.removeAttribute("autoplay");
    v.pause();
    v.controls = true;
  }
} else {
  const visible = new Set();
  // A browser can refuse play() while the page is still loading; retry once the clip has
  // data, then on the first interaction. Show controls only if it is still refused.
  const start = (v) =>
    v.play().catch(() => {
      if (v.dataset.retry) return;
      v.dataset.retry = "1";
      const again = () => {
        if (visible.has(v) && v.paused) v.play().catch(() => (v.controls = true));
      };
      v.addEventListener("canplay", () => setTimeout(again, 300), { once: true });
      setTimeout(again, 1500);
      for (const ev of ["pointerdown", "touchstart", "keydown", "scroll"])
        addEventListener(ev, again, { once: true, passive: true });
    });
  const io = new IntersectionObserver(
    (entries) => {
      for (const e of entries) {
        const v = e.target;
        if (e.isIntersecting) {
          visible.add(v);
          if (v.preload === "none") v.preload = "auto";
          start(v);
        } else {
          visible.delete(v);
          v.pause();
        }
      }
    },
    { threshold: 0.2 }
  );
  for (const v of clips) {
    v.addEventListener("playing", () => (v.controls = false));
    io.observe(v);
  }
}
