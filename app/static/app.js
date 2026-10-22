(function () {
  var reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  // ---------- Mobile nav ----------
  var toggle = document.getElementById("navToggle");
  var navLinks = document.getElementById("navLinks");
  if (toggle && navLinks) {
    toggle.addEventListener("click", function () {
      navLinks.classList.toggle("open");
    });
    navLinks.querySelectorAll("a").forEach(function (a) {
      a.addEventListener("click", function () {
        navLinks.classList.remove("open");
      });
    });
  }

  // ---------- Toast ----------
  window.ppToast = function (msg, ms) {
    var t = document.getElementById("toast");
    if (!t) return;
    t.textContent = msg;
    t.classList.add("show");
    clearTimeout(t._h);
    t._h = setTimeout(function () {
      t.classList.remove("show");
    }, ms || 2600);
  };

  // ---------- Button ripple ----------
  document.addEventListener("click", function (e) {
    var btn = e.target.closest(".btn");
    if (!btn || reduce) return;
    var rect = btn.getBoundingClientRect();
    var d = Math.max(rect.width, rect.height);
    var span = document.createElement("span");
    span.className = "ripple";
    span.style.width = span.style.height = d + "px";
    span.style.left = (e.clientX - rect.left - d / 2) + "px";
    span.style.top = (e.clientY - rect.top - d / 2) + "px";
    btn.appendChild(span);
    setTimeout(function () { span.remove(); }, 650);
  });

  // ---------- Smooth page transition ----------
  function isInternal(a) {
    var href = a.getAttribute("href");
    if (!href || href.charAt(0) === "#" || a.hasAttribute("download")) return false;
    if (a.hasAttribute("data-direct")) return false; // downloads / API target pages
    if (a.target === "_blank") return false;
    try {
      return new URL(a.href, location.href).origin === location.origin;
    } catch (x) {
      return false;
    }
  }
  document.addEventListener("click", function (e) {
    if (e.metaKey || e.ctrlKey || e.shiftKey || e.altKey || reduce) return;
    var a = e.target.closest("a");
    if (!a || !isInternal(a)) return;
    e.preventDefault();
    document.body.classList.remove("entering");
    document.body.classList.add("leaving");
    setTimeout(function () { window.location.href = a.href; }, 170);
  });

  window.addEventListener("pageshow", function () {
    document.body.classList.remove("leaving");
    document.body.classList.add("entering");
    setTimeout(function () { document.body.classList.remove("entering"); }, 480);
  });
  // first paint may already have loaded; force entrance
  if (document.readyState === "complete" || document.readyState === "interactive") {
    document.body.classList.add("entering");
    setTimeout(function () { document.body.classList.remove("entering"); }, 480);
  }

  // ---------- Subtle 3D tilt ----------
  if (!reduce) {
    document.querySelectorAll(".tilt").forEach(function (el) {
      el.addEventListener("mousemove", function (e) {
        var r = el.getBoundingClientRect();
        var px = (e.clientX - r.left) / r.width - 0.5;
        var py = (e.clientY - r.top) / r.height - 0.5;
        el.style.transform =
          "perspective(900px) rotateX(" + (-py * 5).toFixed(2) + "deg) rotateY(" +
          (px * 5).toFixed(2) + "deg) translateZ(8px)";
      });
      el.addEventListener("mouseleave", function () { el.style.transform = ""; });
    });
  }

  // ---------- KPI counters ----------
  document.querySelectorAll("[data-count]").forEach(function (el) {
    var target = parseFloat(el.dataset.count);
    var dec = el.dataset.decimals || 0;
    var ints = dec === "0" || !dec;
    if (!reduce) {
      var step = Math.max(1, target / 40);
      var cur = 0;
      var tick = function () {
        cur += step;
        if (cur >= target) {
          el.textContent = ints ? target.toLocaleString() : target.toFixed(dec);
          return;
        }
        el.textContent = ints ? Math.floor(cur).toLocaleString() : cur.toFixed(dec);
        requestAnimationFrame(tick);
      };
      tick();
    } else {
      el.textContent = ints ? target.toLocaleString() : target.toFixed(dec);
    }
  });

  // ---------- Ctrl/Cmd+S → save the system report ----------
  document.addEventListener("keydown", function (e) {
    if (!(e.ctrlKey || e.metaKey) || e.key.toLowerCase() !== "s") return;
    e.preventDefault();
    if (window.ppToast) window.ppToast("Saving report…");
    fetch("/report/export")
      .then(function (r) {
        if (!r.ok) throw new Error("export failed");
        return r.blob();
      })
      .then(function (blob) {
        var url = URL.createObjectURL(blob);
        var a = document.createElement("a");
        a.href = url;
        a.download = "PlacementPredict_Report.html";
        document.body.appendChild(a);
        a.click();
        a.remove();
        setTimeout(function () { URL.revokeObjectURL(url); }, 4000);
        if (window.ppToast) window.ppToast("Report saved — open the file or print it to PDF");
      })
      .catch(function () {
        if (window.ppToast) window.ppToast("Could not save — try the report page");
      });
  });
})();