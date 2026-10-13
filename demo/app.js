(function () {
  var TOTAL_SCENES = 6;
  var currentScene = 0;
  var activationTimer = null;

  var slider = document.getElementById('slider');
  var progressFill = document.getElementById('progressFill');
  var dots = Array.prototype.slice.call(document.querySelectorAll('.step-dot'));
  var prevBtn = document.getElementById('prevBtn');
  var nextBtn = document.getElementById('nextBtn');
  var nextLabel = document.getElementById('nextLabel');
  var gateSlider = document.getElementById('gateSlider');
  var sampleFold = getSampleFold();

  function sceneEl(index) {
    return document.getElementById('scene-' + index);
  }

  function setDotState() {
    dots.forEach(function (dot, i) {
      dot.classList.remove('active', 'done');
      dot.removeAttribute('aria-current');

      if (i === currentScene) {
        dot.classList.add('active');
        dot.setAttribute('aria-current', 'step');
      } else if (i < currentScene) {
        dot.classList.add('done');
      }
    });
  }

  function resetSceneEffects(nextIndex) {
    if (nextIndex !== 5) {
      var rewardValue = document.querySelector('#scene-5 .reward-value');
      if (rewardValue) {
        var rewardDecimals = Number(rewardValue.getAttribute('data-decimals') || 5);
        rewardValue.textContent = '+' + (0).toFixed(rewardDecimals);
      }

      document.querySelectorAll('#scene-5 .count-up').forEach(function (el) {
        el.textContent = (el.getAttribute('data-prefix') || '') + '0' + (el.getAttribute('data-suffix') || '');
      });
    }
  }

  function goToScene(index) {
    if (index < 0 || index >= TOTAL_SCENES || index === currentScene) return;

    var prev = sceneEl(currentScene);
    if (prev) prev.classList.remove('active');
    resetSceneEffects(index);

    currentScene = index;
    slider.style.transform = 'translateX(-' + (currentScene * 100) + 'vw)';
    progressFill.style.width = ((currentScene + 1) / TOTAL_SCENES * 100) + '%';

    setDotState();
    prevBtn.disabled = currentScene === 0;
    nextLabel.textContent = currentScene === TOTAL_SCENES - 1 ? 'Nuevo ciclo' : 'Siguiente paso';

    window.clearTimeout(activationTimer);
    activationTimer = window.setTimeout(function () {
      var scene = sceneEl(currentScene);
      if (scene) scene.classList.add('active');
      runSceneAnimation(currentScene);
    }, 180);
  }

  function nextScene() {
    goToScene(currentScene === TOTAL_SCENES - 1 ? 0 : currentScene + 1);
  }

  function previousScene() {
    goToScene(currentScene - 1);
  }

  function runSceneAnimation(sceneIndex) {
    if (sceneIndex === 2 && gateSlider) {
      updateGate(Number(gateSlider.value));
    }

    if (sceneIndex === 5) {
      animatePortfolioNumbers();
      animateRewardCounter();
    }
  }

  function setGateOutput(name, text, barValue) {
    var valEl = document.getElementById('gate' + name + 'Val');
    var barEl = document.getElementById('gate' + name + 'Bar');

    if (valEl) valEl.textContent = text;
    if (barEl) barEl.style.width = Math.max(0, Math.min(100, barValue)) + '%';
  }

  function getSampleFold() {
    if (window.DEMO_DATA && Array.isArray(window.DEMO_DATA.folds)) {
      return window.DEMO_DATA.folds.find(function (fold) {
        return fold.fold_id === 5;
      }) || window.DEMO_DATA.folds[0] || null;
    }

    return null;
  }

  function finiteNumber(value) {
    var n = Number(value);
    return Number.isFinite(n) ? n : null;
  }

  function pctText(value, decimals) {
    var n = finiteNumber(value);
    return n === null ? 'n/d' : (n * 100).toFixed(decimals) + '%';
  }

  function setText(id, text) {
    var el = document.getElementById(id);
    if (el) el.textContent = text;
  }

  function setWidth(id, percent) {
    var el = document.getElementById(id);
    if (!el) return;

    var value = finiteNumber(percent);
    el.style.setProperty('--w', Math.max(0, Math.min(100, value === null ? 0 : value)) + '%');
  }

  function setCountTarget(id, target, width) {
    var el = document.getElementById(id);
    if (!el) return;

    var value = finiteNumber(target);
    el.setAttribute('data-target', value === null ? '0' : String(value));
    if (typeof width === 'number') {
      var bar = document.getElementById(id.replace('Metric', 'Bar'));
      if (bar) bar.style.setProperty('--w', Math.max(0, Math.min(100, width)) + '%');
    }
  }

  function updateRealSample() {
    if (!sampleFold) return;

    var toRaw = finiteNumber(sampleFold.to_raw);
    var toExec = finiteNumber(sampleFold.to_exec);
    var grossReturn = finiteNumber(sampleFold.gross_return);
    var netReturn = finiteNumber(sampleFold.net_return);
    var drawdown = finiteNumber(sampleFold.drawdown_incremental);
    var rewardCfg = window.DEMO_DATA && window.DEMO_DATA.config && window.DEMO_DATA.config.environment
      ? window.DEMO_DATA.config.environment.reward || {}
      : {};
    var lambdaTurnover = finiteNumber(rewardCfg.lambda_turnover);
    var lambdaDownside = finiteNumber(rewardCfg.lambda_downside);
    var downsideFactor = finiteNumber(rewardCfg.downside_factor);
    var lambdaDrawdown = finiteNumber(rewardCfg.lambda_drawdown);
    var reward = null;

    if (
      netReturn !== null &&
      toExec !== null &&
      drawdown !== null &&
      lambdaTurnover !== null &&
      lambdaDownside !== null &&
      downsideFactor !== null &&
      lambdaDrawdown !== null
    ) {
      reward = netReturn - lambdaTurnover * toExec - lambdaDrawdown * drawdown;
      if (netReturn < 0) reward -= lambdaDownside * downsideFactor * Math.abs(netReturn);
    }

    setText('rawTurnoverMetric', pctText(toRaw, 1));
    setText('finalTurnoverMetric', pctText(toExec, 1));

    setWidth('stressSignalBar', finiteNumber(sampleFold.stress_signal) === null ? 0 : sampleFold.stress_signal * 100);
    setWidth('volSignalBar', finiteNumber(sampleFold.vol_signal) === null ? 0 : sampleFold.vol_signal * 100);
    setWidth('corrSignalBar', finiteNumber(sampleFold.corr_signal) === null ? 0 : sampleFold.corr_signal * 100);

    setCountTarget('grossReturnMetric', grossReturn === null ? 0 : grossReturn * 100, grossReturn === null ? 0 : Math.min(100, Math.abs(grossReturn) * 22000));
    setCountTarget('netReturnMetric', netReturn === null ? 0 : netReturn * 100, netReturn === null ? 0 : Math.min(100, Math.abs(netReturn) * 30000));
    setCountTarget('portfolioTurnoverMetric', toExec === null ? 0 : toExec * 100, toExec === null ? 0 : toExec / 0.10 * 100);
    setCountTarget('drawdownMetric', drawdown === null ? 0 : drawdown * 100, drawdown === null ? 0 : drawdown * 8500);

    var rewardEl = document.getElementById('rewardMetric');
    if (rewardEl) rewardEl.setAttribute('data-target', reward === null ? '0' : String(reward));
  }

  function updateGate(index) {
    var level = Math.max(0, Math.min(100, Math.round(Number(index) || 0)));
    var gate = level / 100;

    var gateFill = document.getElementById('gateFill');
    var gateHandle = document.getElementById('gateHandle');
    var gateFoldLabel = document.getElementById('gateFoldLabel');

    if (gateFill) gateFill.style.height = level + '%';
    if (gateHandle) gateHandle.style.top = (100 - level) + '%';
    if (gateFoldLabel) {
      gateFoldLabel.textContent = 'contexto aplicado: ' + level + '%';
    }

    setGateOutput('Context', level + '%', level);
    setGateOutput('Fused', 'mercado + ' + level + '% contexto', Math.max(18, level));
  }

  function animateNumber(el, target, duration, formatter) {
    var startTime = null;

    function tick(ts) {
      if (!startTime) startTime = ts;
      var progress = Math.min((ts - startTime) / duration, 1);
      var eased = 1 - Math.pow(1 - progress, 3);
      el.textContent = formatter(eased * target);

      if (progress < 1 && currentScene === 5) {
        window.requestAnimationFrame(tick);
      }
    }

    window.requestAnimationFrame(tick);
  }

  function animatePortfolioNumbers() {
    document.querySelectorAll('#scene-5 .count-up').forEach(function (el) {
      var target = Number(el.getAttribute('data-target') || 0);
      var suffix = el.getAttribute('data-suffix') || '';
      var prefix = el.getAttribute('data-prefix') || '';
      var decimals = Number(el.getAttribute('data-decimals') || 0);
      el.textContent = prefix + '0' + suffix;

      animateNumber(el, target, 950, function (value) {
        var shown = decimals > 0 ? value.toFixed(decimals) : Math.round(value).toString();
        return prefix + shown + suffix;
      });
    });
  }

  function animateRewardCounter() {
    var el = document.querySelector('#scene-5 .reward-value');
    if (!el) return;

    var target = Number(el.getAttribute('data-target') || 0);
    var decimals = Number(el.getAttribute('data-decimals') || 5);
    el.textContent = '+' + (0).toFixed(decimals);

    window.setTimeout(function () {
      if (currentScene !== 5) return;

      animateNumber(el, target, 1200, function (value) {
        return (value >= 0 ? '+' : '') + value.toFixed(decimals);
      });
    }, 900);
  }

  dots.forEach(function (dot) {
    dot.addEventListener('click', function () {
      goToScene(Number(dot.getAttribute('data-i')));
    });
  });

  prevBtn.addEventListener('click', previousScene);
  nextBtn.addEventListener('click', nextScene);

  if (gateSlider) {
    gateSlider.min = 0;
    gateSlider.max = 100;
    gateSlider.step = 1;
    gateSlider.value = 100;
    gateSlider.disabled = false;
    gateSlider.addEventListener('input', function () {
      updateGate(Number(gateSlider.value));
    });
  }

  document.addEventListener('keydown', function (event) {
    if (event.key === 'ArrowRight' || event.key === ' ') {
      event.preventDefault();
      nextScene();
    }

    if (event.key === 'ArrowLeft') {
      event.preventDefault();
      previousScene();
    }
  });

  progressFill.style.width = (1 / TOTAL_SCENES * 100) + '%';
  prevBtn.disabled = true;
  setDotState();
  updateRealSample();
  if (gateSlider) updateGate(Number(gateSlider.value));
  runSceneAnimation(0);
})();
