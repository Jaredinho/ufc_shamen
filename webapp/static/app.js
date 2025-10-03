const form = document.querySelector('#predict-form');
const statusEl = document.querySelector('#status');
const resultsCard = document.querySelector('#results');
const probAEl = document.querySelector('#prob-a');
const probBEl = document.querySelector('#prob-b');
const contribTableBody = document.querySelector('#contributions-table tbody');
const snapshotsEl = document.querySelector('#snapshots');
const fighterInputs = [document.querySelector('#fighter_a'), document.querySelector('#fighter_b')];
const fighterDatalist = document.querySelector('#fighters-list');
let suggestionController = null;
let suggestionTimer = null;

fighterInputs.forEach((input) => {
    if (!input) {
        return;
    }
    input.addEventListener('input', handleSuggestionInput);
    input.addEventListener('focus', handleSuggestionInput);
});

requestFighterSuggestions('');

function handleSuggestionInput(event) {
    if (!fighterDatalist) {
        return;
    }
    const value = event.target.value || '';
    if (suggestionTimer) {
        clearTimeout(suggestionTimer);
    }
    suggestionTimer = setTimeout(() => {
        requestFighterSuggestions(value);
    }, 150);
}

async function requestFighterSuggestions(query) {
    if (!fighterDatalist) {
        return;
    }
    if (suggestionController) {
        suggestionController.abort();
    }
    suggestionController = new AbortController();
    const params = new URLSearchParams();
    if (query && query.trim().length > 0) {
        params.set('q', query.trim());
    }
    params.set('limit', '12');
    try {
        const response = await fetch('/api/suggest_fighters?' + params.toString(), {
            signal: suggestionController.signal,
        });
        if (!response.ok) {
            throw new Error('Suggestion lookup failed');
        }
        const payload = await response.json();
        updateFighterDatalist(Array.isArray(payload.fighters) ? payload.fighters : []);
    } catch (error) {
        if (error.name === 'AbortError') {
            return;
        }
        console.error('Suggestion request failed:', error);
    }
}

function updateFighterDatalist(names) {
    if (!fighterDatalist) {
        return;
    }
    fighterDatalist.innerHTML = '';
    names.forEach((name) => {
        const option = document.createElement('option');
        option.value = name;
        fighterDatalist.appendChild(option);
    });
}

form.addEventListener('submit', async (event) => {
    event.preventDefault();
    statusEl.textContent = 'Scoring matchup…';
    statusEl.classList.remove('error');
    resultsCard.classList.add('hidden');
    contribTableBody.innerHTML = '';
    snapshotsEl.innerHTML = '';

    const formData = new FormData(form);
    const payload = {
        fighter_a: formData.get('fighter_a'),
        fighter_b: formData.get('fighter_b'),
        fight_date: formData.get('fight_date') || undefined,
        weight_class: formData.get('weight_class') || undefined,
        top_contribs: parseInt(formData.get('top_contribs'), 10) || 6,
    };

    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Prediction failed');
        }

        const result = await response.json();
        renderResults(result);
        statusEl.textContent = '';
        resultsCard.classList.remove('hidden');
    } catch (error) {
        statusEl.textContent = error.message;
        statusEl.classList.add('error');
    }
});

function renderResults(data) {
    const { fighter_a, fighter_b, probabilities, contributions, snapshots } = data;

    probAEl.innerHTML = `
        <span class="fighter">${fighter_a}</span>
        <span class="value">${(probabilities[fighter_a] * 100).toFixed(1)}%</span>`;

    probBEl.innerHTML = `
        <span class="fighter">${fighter_b}</span>
        <span class="value">${(probabilities[fighter_b] * 100).toFixed(1)}%</span>`;

    contributions.forEach(({ feature, log_odds_contribution }) => {
        const row = document.createElement('tr');
        const featureCell = document.createElement('td');
        const impactCell = document.createElement('td');
        featureCell.textContent = feature;
        impactCell.textContent = `${log_odds_contribution >= 0 ? '+' : ''}${log_odds_contribution.toFixed(4)}`;
        row.append(featureCell, impactCell);
        contribTableBody.appendChild(row);
    });

    snapshots.forEach((snapshot) => {
        const card = document.createElement('div');
        card.className = 'snapshot-card';
        card.innerHTML = `
            <h4>${snapshot.fighter_name}</h4>
            <div class="snapshot-grid">
                <span>Weight Class</span><strong>${snapshot.weight_class}</strong>
                <span>Stance</span><strong>${snapshot.stance}</strong>
                <span>Height</span><strong>${snapshot.height}</strong>
                <span>Reach</span><strong>${snapshot.reach}</strong>
                <span>Weight</span><strong>${snapshot.weight}</strong>
                <span>Age</span><strong>${snapshot.age}</strong>
                <span>Recorded Fights</span><strong>${snapshot.total_fights}</strong>
                <span>Win Rate</span><strong>${snapshot.win_rate}</strong>
                <span>Last Layoff</span><strong>${snapshot.days_since_last_fight}</strong>
                <span>Sig. Strikes Avg</span><strong>${snapshot.sig_strikes_avg}</strong>
                <span>Sig. Strikes Acc.</span><strong>${snapshot.sig_strikes_acc}</strong>
                <span>Takedowns Avg</span><strong>${snapshot.takedowns_avg}</strong>
                <span>Takedown Acc.</span><strong>${snapshot.takedown_acc}</strong>
                <span>Control Time Avg</span><strong>${snapshot.control_time_avg}</strong>
                <span>Sub Attempts Avg</span><strong>${snapshot.sub_attempts_avg}</strong>
            </div>`;
        snapshotsEl.appendChild(card);
    });
}
