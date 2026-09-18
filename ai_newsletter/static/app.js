document.addEventListener("DOMContentLoaded", () => {
  if (window.lucide) {
    window.lucide.createIcons();
  }

  const status = document.querySelector("[data-status]");
  const collectForm = document.querySelector("[data-collect-form]");
  const sendForm = document.querySelector("[data-send-form]");
  const mailSettingsForm = document.querySelector("[data-mail-settings-form]");
  const passwordState = document.querySelector("[data-password-state]");
  const regionButtons = Array.from(document.querySelectorAll("[data-region-filter]"));
  const topicButtons = Array.from(document.querySelectorAll("[data-topic-filter]"));
  const sourceTypeButtons = Array.from(document.querySelectorAll("[data-source-type-filter]"));
  const issueSections = Array.from(document.querySelectorAll(".issue-section:not(.table-briefing-section)"));
  const filterEmpty = document.querySelector("[data-filter-empty]");
  const viewTabs = Array.from(document.querySelectorAll("[data-view-tab]"));
  const viewPanels = Array.from(document.querySelectorAll("[data-view-panel]"));
  const mailSettingsModal = document.getElementById("mailSettingsModal");
  const openMailSettingsButtons = Array.from(document.querySelectorAll("[data-open-mail-settings]"));
  const closeMailSettingsButtons = Array.from(document.querySelectorAll("[data-close-mail-settings]"));
  const sourceHealthAction = document.querySelector("[data-source-health-action]");
  const sourceHealthSummary = document.querySelector("[data-source-health-summary]");
  const sourceHealthList = document.querySelector("[data-source-health-list]");

  const showStatus = (message, tone = "neutral") => {
    if (!status) return;
    status.textContent = message;
    status.dataset.tone = tone;
  };

  const getAdminKey = () => localStorage.getItem("admin_key") || "";

  const adminHeaders = () => {
    const key = getAdminKey();
    return key ? { "X-Admin-Key": key, "Authorization": `Bearer ${key}` } : {};
  };

  const checkAdminMode = () => {
    const key = getAdminKey();
    if (key) {
      document.body.classList.add("admin-mode");
    } else {
      document.body.classList.remove("admin-mode");
    }
  };

  checkAdminMode();

  // Secret trigger: clicking .brand-mark 5 times
  const brandMark = document.querySelector(".brand-mark");
  if (brandMark) {
    let clickCount = 0;
    let clickTimer = null;
    brandMark.addEventListener("click", () => {
      clickCount += 1;
      clearTimeout(clickTimer);
      clickTimer = setTimeout(() => {
        clickCount = 0;
      }, 2500);

      if (clickCount >= 5) {
        clickCount = 0;
        const currentKey = getAdminKey();
        if (currentKey) {
          if (confirm("관리자 모드를 종료(로그아웃)하시겠습니까?")) {
            localStorage.removeItem("admin_key");
            checkAdminMode();
            showStatus("관리자 모드가 해제되었습니다.");
          }
        } else {
          const input = prompt("관리자 키(비밀번호)를 입력하세요:");
          if (input !== null && input.trim()) {
            localStorage.setItem("admin_key", input.trim());
            checkAdminMode();
            showStatus("관리자 모드가 활성화되었습니다.");
          }
        }
      }
    });
  }

  if (collectForm) {
    collectForm.addEventListener("submit", async (event) => {
      event.preventDefault();
      showStatus("AI 뉴스를 수집하고 있습니다. 잠시만 기다려 주세요...");
      const button = collectForm.querySelector("button");
      button.disabled = true;
      try {
        const response = await fetch("/api/collect", {
          method: "POST",
          headers: adminHeaders(),
        });
        const payload = await response.json();
        if (!response.ok || !payload.ok) {
          if (response.status === 401) {
            localStorage.removeItem("admin_key");
            checkAdminMode();
          }
          throw new Error(payload.message || "수집에 실패했습니다.");
        }
        window.location.href = `/issues/${payload.issue_date}`;
      } catch (error) {
        showStatus(error.message, "error");
      } finally {
        button.disabled = false;
      }
    });
  }

  // Filtering logic
  const sectionButtons = Array.from(document.querySelectorAll("[data-section-filter]"));
  const metricCards = Array.from(document.querySelectorAll(".metric-card[data-priority-filter]"));
  const briefingTable = document.getElementById("briefingTable");
  const tableSearchInput = document.getElementById("briefingTableSearch");
  const tableFilterCategory = document.getElementById("tableFilterCategory");
  const tableFilterSource = document.getElementById("tableFilterSource");
  const tableFilterConfidence = document.getElementById("tableFilterConfidence");
  const tablePageSize = document.getElementById("tablePageSize");
  const tableFilterReset = document.getElementById("tableFilterReset");
  const tableFilterCount = document.getElementById("tableFilterCount");
  const tablePaginationInfo = document.getElementById("tablePaginationInfo");
  const tablePaginationControls = document.getElementById("tablePaginationControls");

  let activeSection = "all";
  let activeRegion = "all";
  let activeTopic = "all";
  let activeSourceType = "all";
  let activePriority = "all";

  // ── Table State & Management (Sorting, Filtering, Pagination) ───────────────
  let currentSortCol = "priority";
  let currentSortDir = "desc";
  let currentPage = 1;
  let pageSize = 10;

  const initTableManager = () => {
    if (!briefingTable) return;

    const tbody = briefingTable.querySelector("tbody");
    if (!tbody) return;

    const allRows = Array.from(tbody.querySelectorAll("tr"));
    const sortHeaders = Array.from(briefingTable.querySelectorAll("th.sortable-th"));

    const updateSortIcons = () => {
      sortHeaders.forEach((th) => {
        const col = th.dataset.sort;
        const icon = th.querySelector("[data-sort-icon]");
        if (col === currentSortCol) {
          th.classList.add("sort-active");
          if (icon) icon.textContent = currentSortDir === "asc" ? "▲" : "▼";
        } else {
          th.classList.remove("sort-active");
          if (icon) icon.textContent = "↕";
        }
      });
    };

    const renderTable = () => {
      const categoryVal = tableFilterCategory ? tableFilterCategory.value : "all";
      const sourceVal = tableFilterSource ? tableFilterSource.value : "all";
      const confidenceVal = tableFilterConfidence ? tableFilterConfidence.value : "all";
      const query = tableSearchInput ? tableSearchInput.value.toLowerCase().trim() : "";

      // 1. Filter rows
      const filteredRows = allRows.filter((row) => {
        const rowPriority = row.dataset.priority || "";
        const rowCategory = row.dataset.category || "";
        const rowSource = row.dataset.source || "";
        const rowConfidence = row.dataset.confidence || "";

        const priorityMatch = activePriority === "all" || rowPriority === activePriority;
        const categoryMatch = categoryVal === "all" || rowCategory === categoryVal;
        const sourceMatch = sourceVal === "all" || rowSource === sourceVal;
        const confidenceMatch = confidenceVal === "all" || rowConfidence === confidenceVal;
        const searchMatch = !query || row.textContent.toLowerCase().includes(query);

        return priorityMatch && categoryMatch && sourceMatch && confidenceMatch && searchMatch;
      });

      // 2. Sort rows
      filteredRows.sort((a, b) => {
        let valA, valB;
        if (currentSortCol === "priority") {
          valA = parseFloat(a.dataset.score) || 0;
          valB = parseFloat(b.dataset.score) || 0;
          return currentSortDir === "asc" ? valA - valB : valB - valA;
        } else if (currentSortCol === "confidence") {
          valA = parseInt(a.dataset.confidenceRank, 10) || 0;
          valB = parseInt(b.dataset.confidenceRank, 10) || 0;
          return currentSortDir === "asc" ? valA - valB : valB - valA;
        } else if (currentSortCol === "date") {
          valA = a.dataset.date || "";
          valB = b.dataset.date || "";
          return currentSortDir === "asc" ? valA.localeCompare(valB) : valB.localeCompare(valA);
        } else if (currentSortCol === "category") {
          valA = a.dataset.category || "";
          valB = b.dataset.category || "";
          return currentSortDir === "asc" ? valA.localeCompare(valB) : valB.localeCompare(valA);
        } else if (currentSortCol === "source") {
          valA = a.dataset.source || "";
          valB = b.dataset.source || "";
          return currentSortDir === "asc" ? valA.localeCompare(valB) : valB.localeCompare(valA);
        } else {
          // title
          valA = a.dataset.title || "";
          valB = b.dataset.title || "";
          return currentSortDir === "asc" ? valA.localeCompare(valB) : valB.localeCompare(valA);
        }
      });

      // 3. Pagination calculation
      const totalFiltered = filteredRows.length;
      const effectivePageSize = pageSize === "all" ? (totalFiltered || 1) : parseInt(pageSize, 10);
      const totalPages = Math.max(1, Math.ceil(totalFiltered / effectivePageSize));
      if (currentPage > totalPages) currentPage = totalPages;
      if (currentPage < 1) currentPage = 1;

      const startIndex = (currentPage - 1) * effectivePageSize;
      const endIndex = Math.min(startIndex + effectivePageSize, totalFiltered);

      // Hide all rows, re-append sorted, show page slice
      allRows.forEach((r) => (r.style.display = "none"));
      filteredRows.forEach((row, idx) => {
        tbody.appendChild(row);
        if (idx >= startIndex && idx < endIndex) {
          row.style.display = "";
        } else {
          row.style.display = "none";
        }
      });

      // 4. Update info counts
      const isKo = !document.documentElement.classList.contains("lang-en-active");
      if (tableFilterCount) {
        tableFilterCount.textContent = isKo ? `총 ${totalFiltered}건` : `Total ${totalFiltered}`;
      }

      if (tablePaginationInfo) {
        if (totalFiltered === 0) {
          tablePaginationInfo.textContent = isKo ? "검색 결과 0건" : "0 results";
        } else {
          tablePaginationInfo.textContent = isKo
            ? `${startIndex + 1}-${endIndex} / 전체 ${totalFiltered}건`
            : `${startIndex + 1}-${endIndex} of ${totalFiltered}`;
        }
      }

      // 5. Render page buttons
      if (tablePaginationControls) {
        tablePaginationControls.innerHTML = "";
        if (totalPages > 1) {
          const prevBtn = document.createElement("button");
          prevBtn.type = "button";
          prevBtn.className = "page-btn";
          prevBtn.textContent = "«";
          prevBtn.disabled = currentPage === 1;
          prevBtn.title = "이전 페이지";
          prevBtn.addEventListener("click", () => {
            if (currentPage > 1) {
              currentPage -= 1;
              renderTable();
            }
          });
          tablePaginationControls.appendChild(prevBtn);

          let startPage = Math.max(1, currentPage - 2);
          let endPage = Math.min(totalPages, currentPage + 2);
          if (currentPage <= 3) {
            endPage = Math.min(totalPages, 5);
          } else if (currentPage >= totalPages - 2) {
            startPage = Math.max(1, totalPages - 4);
          }

          for (let p = startPage; p <= endPage; p++) {
            const pageBtn = document.createElement("button");
            pageBtn.type = "button";
            pageBtn.className = `page-btn ${p === currentPage ? "active" : ""}`;
            pageBtn.textContent = p;
            pageBtn.addEventListener("click", () => {
              currentPage = p;
              renderTable();
            });
            tablePaginationControls.appendChild(pageBtn);
          }

          const nextBtn = document.createElement("button");
          nextBtn.type = "button";
          nextBtn.className = "page-btn";
          nextBtn.textContent = "»";
          nextBtn.disabled = currentPage === totalPages;
          nextBtn.title = "다음 페이지";
          nextBtn.addEventListener("click", () => {
            if (currentPage < totalPages) {
              currentPage += 1;
              renderTable();
            }
          });
          tablePaginationControls.appendChild(nextBtn);
        }
      }

      updateSortIcons();
    };

    // Sort Click Handlers
    sortHeaders.forEach((th) => {
      const col = th.dataset.sort;
      const handleSort = () => {
        if (currentSortCol === col) {
          currentSortDir = currentSortDir === "asc" ? "desc" : "asc";
        } else {
          currentSortCol = col;
          currentSortDir = (col === "priority" || col === "confidence" || col === "date") ? "desc" : "asc";
        }
        currentPage = 1;
        renderTable();
      };
      th.addEventListener("click", handleSort);
      th.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          handleSort();
        }
      });
    });

    // Filter Change Listeners
    if (tableFilterCategory) {
      tableFilterCategory.addEventListener("change", () => {
        currentPage = 1;
        renderTable();
      });
    }
    if (tableFilterSource) {
      tableFilterSource.addEventListener("change", () => {
        currentPage = 1;
        renderTable();
      });
    }
    if (tableFilterConfidence) {
      tableFilterConfidence.addEventListener("change", () => {
        currentPage = 1;
        renderTable();
      });
    }
    if (tableSearchInput) {
      tableSearchInput.addEventListener("input", () => {
        currentPage = 1;
        renderTable();
      });
    }
    if (tablePageSize) {
      tablePageSize.addEventListener("change", () => {
        pageSize = tablePageSize.value;
        currentPage = 1;
        renderTable();
      });
    }
    if (tableFilterReset) {
      tableFilterReset.addEventListener("click", () => {
        if (tableFilterCategory) tableFilterCategory.value = "all";
        if (tableFilterSource) tableFilterSource.value = "all";
        if (tableFilterConfidence) tableFilterConfidence.value = "all";
        if (tableSearchInput) tableSearchInput.value = "";
        if (tablePageSize) {
          tablePageSize.value = "10";
          pageSize = 10;
        }
        currentSortCol = "priority";
        currentSortDir = "desc";
        currentPage = 1;
        renderTable();
      });
    }

    window._renderBriefingTable = renderTable;
    renderTable();
  };

  const applyFilters = () => {
    let totalVisible = 0;

    issueSections.forEach((section) => {
      const secKey = section.dataset.sectionKey || "";
      const sectionMatchOverall = activeSection === "all" || activeSection === secKey;

      const cards = Array.from(section.querySelectorAll(".article-card"));
      let visibleInSection = 0;

      cards.forEach((card) => {
        const cardSec = card.dataset.section || secKey;
        const regions = (card.dataset.regions || "").split(" ");
        const topics = (card.dataset.topics || "").split(" ");
        const sourceTypes = (card.dataset.sourceType || "").split(" ");
        const cardPriority = card.dataset.priority || "";

        const sectionMatch = activeSection === "all" || cardSec === activeSection;
        const regionMatch = activeRegion === "all" || regions.includes(activeRegion);
        const topicMatch = activeTopic === "all" || topics.includes(activeTopic);
        const sourceMatch = activeSourceType === "all" || sourceTypes.includes(activeSourceType);
        const priorityMatch = activePriority === "all" || cardPriority === activePriority;

        if (sectionMatch && regionMatch && topicMatch && sourceMatch && priorityMatch) {
          card.style.display = "";
          visibleInSection += 1;
        } else {
          card.style.display = "none";
        }
      });

      if (sectionMatchOverall && visibleInSection > 0) {
        section.style.display = "";
        totalVisible += visibleInSection;
      } else {
        section.style.display = "none";
      }
    });

    if (filterEmpty) {
      filterEmpty.style.display = totalVisible === 0 ? "block" : "none";
    }

    if (window._renderBriefingTable) {
      window._renderBriefingTable();
    }
  };

  metricCards.forEach((card) => {
    const handlePriorityClick = () => {
      const filterVal = card.dataset.priorityFilter;
      if (filterVal === activePriority && filterVal !== "all") {
        // Toggle back to 'all' if already selected
        activePriority = "all";
      } else {
        activePriority = filterVal;
      }

      metricCards.forEach((c) => {
        if (c.dataset.priorityFilter === activePriority) {
          c.classList.add("active");
        } else {
          c.classList.remove("active");
        }
      });

      applyFilters();
    };

    card.addEventListener("click", handlePriorityClick);
    card.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        handlePriorityClick();
      }
    });
  });

  sectionButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      sectionButtons.forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      activeSection = btn.dataset.sectionFilter;
      applyFilters();
    });
  });

  regionButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      regionButtons.forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      activeRegion = btn.dataset.regionFilter;
      applyFilters();
    });
  });

  topicButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      topicButtons.forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      activeTopic = btn.dataset.topicFilter;
      applyFilters();
    });
  });

  sourceTypeButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      sourceTypeButtons.forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      activeSourceType = btn.dataset.sourceTypeFilter;
      applyFilters();
    });
  });

  // View tab navigation
  viewTabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      const target = tab.dataset.viewTab;
      viewTabs.forEach((t) => t.classList.remove("active"));
      tab.classList.add("active");

      viewPanels.forEach((p) => {
        if (p.dataset.viewPanel === target) {
          p.style.display = "";
        } else {
          p.style.display = "none";
        }
      });

      if (target === "sources") {
        loadSourceHealth();
      }
    });
  });

  const openMailSettings = async () => {
    let key = getAdminKey();
    if (!key) {
      const input = prompt("관리자 암호(비밀번호)를 입력하세요:");
      if (!input || !input.trim()) return;
      localStorage.setItem("admin_key", input.trim());
      checkAdminMode();
    }
    await loadMailSettings();
    if (mailSettingsModal) {
      mailSettingsModal.style.display = "flex";
      if (window.lucide) {
        window.lucide.createIcons();
      }
    }
  };

  const closeMailSettings = () => {
    if (mailSettingsModal) {
      mailSettingsModal.style.display = "none";
    }
  };

  openMailSettingsButtons.forEach((btn) => {
    btn.addEventListener("click", (e) => {
      e.preventDefault();
      openMailSettings();
    });
  });

  closeMailSettingsButtons.forEach((btn) => {
    btn.addEventListener("click", (e) => {
      e.preventDefault();
      closeMailSettings();
    });
  });

  if (mailSettingsModal) {
    mailSettingsModal.addEventListener("click", (e) => {
      if (e.target === mailSettingsModal) {
        closeMailSettings();
      }
    });
  }

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && mailSettingsModal && mailSettingsModal.style.display !== "none") {
      closeMailSettings();
    }
  });

  const loadMailSettings = async () => {
    try {
      const res = await fetch("/api/mail-settings", { headers: adminHeaders() });
      if (res.status === 401) {
        const input = prompt("관리자 암호가 일치하지 않습니다. 다시 입력하세요:");
        if (input && input.trim()) {
          localStorage.setItem("admin_key", input.trim());
          checkAdminMode();
          return loadMailSettings();
        } else {
          return;
        }
      }
      if (!res.ok) return;
      const data = await res.json();
      if (!data.ok || !data.settings) return;
      const s = data.settings;
      if (mailSettingsForm) {
        mailSettingsForm.smtp_host.value = s.smtp_host || "";
        mailSettingsForm.smtp_port.value = s.smtp_port || 587;
        mailSettingsForm.smtp_user.value = s.smtp_user || "";
        mailSettingsForm.smtp_from.value = s.smtp_from || "";
        mailSettingsForm.newsletter_to.value = s.newsletter_to || "";
        mailSettingsForm.smtp_tls.checked = Boolean(s.smtp_tls);
        if (passwordState) {
          passwordState.textContent = s.smtp_password_saved ? "설정됨 (암호화 보관)" : "미설정";
        }
      }
    } catch (e) {}
  };

  if (mailSettingsForm) {
    mailSettingsForm.addEventListener("submit", async (e) => {
      e.preventDefault();
      const payload = {
        smtp_host: mailSettingsForm.smtp_host.value.trim(),
        smtp_port: parseInt(mailSettingsForm.smtp_port.value, 10) || 587,
        smtp_user: mailSettingsForm.smtp_user.value.trim(),
        smtp_password: mailSettingsForm.smtp_password.value,
        smtp_from: mailSettingsForm.smtp_from.value.trim(),
        newsletter_to: mailSettingsForm.newsletter_to.value.trim(),
        smtp_tls: mailSettingsForm.smtp_tls.checked,
      };
      try {
        const res = await fetch("/api/mail-settings", {
          method: "POST",
          headers: { "Content-Type": "application/json", ...adminHeaders() },
          body: JSON.stringify(payload),
        });
        const data = await res.json();
        if (!res.ok || !data.ok) throw new Error(data.message || "설정 저장 실패");
        showStatus(data.message || "설정이 저장되었습니다.", "success");
        await loadMailSettings();
        setTimeout(() => {
          closeMailSettings();
        }, 500);
      } catch (err) {
        showStatus(err.message, "error");
      }
    });
  }

  const loadSourceHealth = async () => {
    if (!sourceHealthList) return;
    sourceHealthList.innerHTML = '<p class="empty-note">소스 피드 연결 상태를 확인하고 있습니다...</p>';
    try {
      const res = await fetch("/api/sources/health", { headers: adminHeaders() });
      if (!res.ok) {
        sourceHealthList.innerHTML = '<p class="empty-note">관리자 권한이 필요합니다.</p>';
        return;
      }
      const data = await res.json();
      if (!data.ok || !data.sources) return;

      const sources = data.sources;
      const okCount = sources.filter((s) => s.status === "ok").length;
      if (sourceHealthSummary) {
        sourceHealthSummary.textContent = `전체 ${sources.length}개 소스 중 ${okCount}개 정상 작동 중`;
      }

      sourceHealthList.innerHTML = sources
        .map(
          (s) => `
        <div class="source-health-row">
          <div>
            <div style="font-weight:700;font-size:14px;color:#0f172a;">${s.name}</div>
            <div style="font-size:12px;color:#64748b;">${s.id} · ${s.entries_count}건 수집</div>
            ${s.error ? `<div style="font-size:12px;color:#ef4444;margin-top:2px;">오류: ${s.error}</div>` : ""}
          </div>
          <div>
            <span class="source-status-badge ${s.status === "ok" ? "ok" : "err"}">
              ${s.status === "ok" ? "정상" : "오류"}
            </span>
          </div>
        </div>
      `
        )
        .join("");
    } catch (e) {
      sourceHealthList.innerHTML = '<p class="empty-note">피드 상태 조회 중 오류 발생</p>';
    }
  };

  if (sourceHealthAction) {
    sourceHealthAction.addEventListener("click", loadSourceHealth);
  }

  // Send email form
  if (sendForm) {
    sendForm.addEventListener("submit", async (e) => {
      e.preventDefault();
      const issueDate = sendForm.dataset.issueDate;
      const isEn = document.documentElement.classList.contains("lang-en-active");
      const lang = isEn ? "en" : "ko";

      if (!confirm(`${issueDate} AI 뉴스레터(${lang.toUpperCase()} 버전)를 발송하시겠습니까?`)) {
        return;
      }

      showStatus("이메일을 발송하고 있습니다...");
      const btn = sendForm.querySelector("button");
      btn.disabled = true;

      try {
        const res = await fetch(`/api/issues/${issueDate}/send?lang=${lang}`, {
          method: "POST",
          headers: adminHeaders(),
        });
        const data = await res.json();
        if (!res.ok || !data.ok) {
          throw new Error(data.message || "발송 실패");
        }
        showStatus(data.message || "이메일이 성공적으로 발송되었습니다.", "success");
      } catch (err) {
        showStatus(err.message, "error");
      } finally {
        btn.disabled = false;
      }
    });
  }

  // Language Toggle (KO <-> EN)
  const langToggle = document.querySelector("[data-lang-toggle]");
  if (langToggle) {
    const textSpan = langToggle.querySelector(".lang-text");
    const savedLang = localStorage.getItem("preferred_lang");

    if (savedLang === "en") {
      document.documentElement.classList.add("lang-en-active");
      if (textSpan) textSpan.textContent = "KO";
    }

    langToggle.addEventListener("click", () => {
      document.documentElement.classList.toggle("lang-en-active");
      const isEn = document.documentElement.classList.contains("lang-en-active");
      if (textSpan) {
        textSpan.textContent = isEn ? "KO" : "EN";
      }
      localStorage.setItem("preferred_lang", isEn ? "en" : "ko");
    });
  }

  // Theme Toggle (Light <-> Night / Dark Mode)
  const themeToggle = document.querySelector("[data-theme-toggle]");
  if (themeToggle) {
    const updateThemeIcon = (isDark) => {
      const moon = themeToggle.querySelector(".icon-moon");
      const sun = themeToggle.querySelector(".icon-sun");
      if (moon && sun) {
        moon.style.display = isDark ? "none" : "inline-block";
        sun.style.display = isDark ? "inline-block" : "none";
      }
    };

    const isCurrentDark = document.documentElement.classList.contains("dark-mode");
    updateThemeIcon(isCurrentDark);

    themeToggle.addEventListener("click", () => {
      const nowDark = document.documentElement.classList.toggle("dark-mode");
      updateThemeIcon(nowDark);
      localStorage.setItem("theme", nowDark ? "dark" : "light");
      if (window.lucide) {
        window.lucide.createIcons();
      }
    });
  }

  // AdSense active container observer
  const checkAdContainers = () => {
    document.querySelectorAll(".ad-container").forEach((container) => {
      const ins = container.querySelector("ins.adsbygoogle");
      if (ins) {
        const hasIframe = ins.querySelector("iframe") !== null;
        const isFilled = ins.getAttribute("data-ad-status") === "filled";
        if (hasIframe || isFilled) {
          container.classList.add("is-active");
        } else {
          container.classList.remove("is-active");
        }
      }
    });
  };

  checkAdContainers();
  // Periodically check if AdSense loaded
  setTimeout(checkAdContainers, 1500);
  setTimeout(checkAdContainers, 3500);

  // Initialize briefing table sorting, filtering, and pagination
  initTableManager();
});


