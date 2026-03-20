<script lang="ts">
	import type { Collection, User, ContentImage, Location } from '$lib/types';
	import { onMount } from 'svelte';
	import { t } from 'svelte-i18n';
	import { DefaultMarker, MapLibre, Popup } from 'svelte-maplibre';
	import { getBasemapUrl } from '$lib';
	import MagnifyIcon from '~icons/mdi/magnify';
	import MapMarker from '~icons/mdi/map-marker';
	import Star from '~icons/mdi/star';
	import StarHalfFull from '~icons/mdi/star-half-full';
	import StarOutline from '~icons/mdi/star-outline';
	import TuneVariant from '~icons/mdi/tune-variant';
	import CloseCircle from '~icons/mdi/close-circle';
	import RobotOutline from '~icons/mdi/robot-outline';
	import LocationModal from '$lib/components/locations/LocationModal.svelte';
	import { createEventDispatcher } from 'svelte';

	export let collection: Collection;
	export let user: User | null;

	type GemCard = {
		gmap_id: string;
		name: string;
		latitude: number;
		longitude: number;
		score: number;
		confidence_pct: string;
		vibe_tags: string[];
		category: string;
		distance_km: number;
		avg_rating: number;
		num_reviews: number;
	};

	let selectedLocationId: string | null = null;
	let loading = false;
	let gemCards: GemCard[] = [];
	let error: string | null = null;
	let radiusKm = 10;
	let topK = 10;
	let mapCenter: { lng: number; lat: number } = { lng: 0, lat: 0 };
	let mapZoom = 12;
	let mlServiceHealth: 'unknown' | 'healthy' | 'unreachable' = 'unknown';
	let modelVersion = '';
	let inferenceTimeMs = 0;
	let candidatesScored = 0;
	let showFilters = false;

	// Modal for creating location from gem card
	let showLocationModal = false;
	let modalLocationToEdit: Location | null = null;

	const dispatch = createEventDispatcher();

	// Get locations with coordinates
	$: locationsWithCoords = collection.locations.filter((l) => l.latitude && l.longitude);

	$: isMetric = user?.measurement_system === 'metric';
	$: radiusDisplay = isMetric ? `${radiusKm} km` : `${(radiusKm / 1.60934).toFixed(1)} mi`;

	onMount(async () => {
		if (locationsWithCoords.length > 0) {
			selectedLocationId = locationsWithCoords[0].id;
			mapCenter = {
				lng: locationsWithCoords[0].longitude!,
				lat: locationsWithCoords[0].latitude!
			};
		}
		// Check ML service health
		try {
			const resp = await fetch('/api/gemspot/health/');
			if (resp.ok) {
				const data = await resp.json();
				mlServiceHealth = data.status === 'healthy' ? 'healthy' : 'unreachable';
				modelVersion = data.model_version || '';
			} else {
				mlServiceHealth = 'unreachable';
			}
		} catch {
			mlServiceHealth = 'unreachable';
		}
	});

	// Update map center when selected location changes
	$: if (selectedLocationId) {
		const location = locationsWithCoords.find((l) => l.id === selectedLocationId);
		if (location && location.latitude && location.longitude) {
			mapCenter = { lng: location.longitude, lat: location.latitude };
		}
	}

	async function searchGemSpot() {
		if (!selectedLocationId) {
			error = 'Please select a location to search around';
			return;
		}

		const location = locationsWithCoords.find((l) => l.id === selectedLocationId);
		if (!location || !location.latitude || !location.longitude) {
			error = 'Selected location has no coordinates';
			return;
		}

		loading = true;
		error = null;
		gemCards = [];

		try {
			const params = new URLSearchParams({
				lat: location.latitude.toString(),
				lon: location.longitude.toString(),
				radius_km: radiusKm.toString(),
				top_k: topK.toString()
			});

			const response = await fetch(`/api/gemspot/?${params.toString()}`);

			if (!response.ok) {
				const errorData = await response.json();
				throw new Error(errorData.error || 'Failed to fetch GemSpot recommendations');
			}

			const data = await response.json();

			if (data.error) {
				throw new Error(data.error);
			}

			gemCards = data.gem_cards || [];
			modelVersion = data.model_version || '';
			inferenceTimeMs = data.inference_time_ms || 0;
			candidatesScored = data.candidates_scored || 0;

			if (gemCards.length > 0) {
				const lats = gemCards.map((g) => g.latitude);
				const lngs = gemCards.map((g) => g.longitude);
				mapCenter = {
					lat: lats.reduce((a, b) => a + b, 0) / lats.length,
					lng: lngs.reduce((a, b) => a + b, 0) / lngs.length
				};
			}
		} catch (err) {
			error = err instanceof Error ? err.message : 'An error occurred';
			console.error('GemSpot error:', err);
		} finally {
			loading = false;
		}
	}

	function getScoreColor(score: number): string {
		if (score >= 0.8) return 'badge-success';
		if (score >= 0.6) return 'badge-warning';
		return 'badge-error';
	}

	function getVibeTagColor(tag: string): string {
		const colors: Record<string, string> = {
			scenic: 'bg-emerald-500/20 text-emerald-300 border-emerald-500/30',
			relaxing: 'bg-blue-500/20 text-blue-300 border-blue-500/30',
			adventurous: 'bg-orange-500/20 text-orange-300 border-orange-500/30',
			cultural: 'bg-purple-500/20 text-purple-300 border-purple-500/30',
			romantic: 'bg-pink-500/20 text-pink-300 border-pink-500/30',
			'family-friendly': 'bg-yellow-500/20 text-yellow-300 border-yellow-500/30',
			nightlife: 'bg-indigo-500/20 text-indigo-300 border-indigo-500/30',
			foodie: 'bg-red-500/20 text-red-300 border-red-500/30',
			historic: 'bg-amber-500/20 text-amber-300 border-amber-500/30',
			trendy: 'bg-cyan-500/20 text-cyan-300 border-cyan-500/30'
		};
		return colors[tag] || 'bg-base-300 text-base-content border-base-content/20';
	}

	function formatDistance(km: number) {
		if (isMetric) {
			return km < 1 ? `${Math.round(km * 1000)} m` : `${km.toFixed(1)} km`;
		} else {
			const miles = km / 1.60934;
			const feet = miles * 5280;
			return miles < 0.1 ? `${Math.round(feet)} ft` : `${miles.toFixed(1)} mi`;
		}
	}

	function renderStars(rating: number | undefined) {
		if (!rating) return [];
		const stars = [];
		const fullStars = Math.floor(rating);
		const hasHalfStar = rating % 1 >= 0.5;
		for (let i = 0; i < 5; i++) {
			if (i < fullStars) stars.push({ type: 'full', key: i });
			else if (i === fullStars && hasHalfStar) stars.push({ type: 'half', key: i });
			else stars.push({ type: 'empty', key: i });
		}
		return stars;
	}

	function openCreateLocationFromGemCard(card: GemCard) {
		modalLocationToEdit = {
			id: '',
			name: card.name,
			location: '',
			tags: card.vibe_tags,
			description: `AI Confidence: ${card.confidence_pct} | Category: ${card.category}`,
			rating: card.avg_rating || NaN,
			price: null,
			price_currency: null,
			link: null,
			images: [],
			visits: [],
			collections: [collection.id],
			latitude: card.latitude,
			longitude: card.longitude,
			is_public: false,
			user: user ?? null,
			category: null,
			attachments: [],
			trails: []
		} as Location;
		showLocationModal = true;
	}

	function handleLocationCreate(e: CustomEvent) {
		showLocationModal = false;
		modalLocationToEdit = null;
		collection.locations = [...collection.locations, e.detail];
	}
</script>

{#if showLocationModal}
	<LocationModal
		{user}
		{collection}
		locationToEdit={modalLocationToEdit}
		on:create={handleLocationCreate}
		on:save={handleLocationCreate}
		on:close={() => {
			showLocationModal = false;
			modalLocationToEdit = null;
		}}
	/>
{/if}

<div class="space-y-6">
	<!-- Header Card with AI branding -->
	<div class="card bg-gradient-to-br from-primary/10 via-base-200 to-secondary/10 shadow-xl border border-primary/20">
		<div class="card-body">
			<div class="flex items-center gap-3 mb-4">
				<div
					class="w-12 h-12 rounded-xl bg-gradient-to-br from-primary to-secondary flex items-center justify-center shadow-lg"
				>
					<RobotOutline class="w-7 h-7 text-primary-content" />
				</div>
				<div>
					<h2 class="card-title text-2xl">GemSpot AI</h2>
					<p class="text-sm opacity-70">ML-powered hidden gem recommendations</p>
				</div>
				<!-- Health indicator -->
				<div class="ml-auto">
					{#if mlServiceHealth === 'healthy'}
						<div class="badge badge-success gap-1">
							<span class="w-2 h-2 rounded-full bg-success-content animate-pulse"></span>
							Online
						</div>
					{:else if mlServiceHealth === 'unreachable'}
						<div class="badge badge-error gap-1">
							<span class="w-2 h-2 rounded-full bg-error-content"></span>
							Offline
						</div>
					{:else}
						<div class="badge badge-ghost gap-1">
							<span class="w-2 h-2 rounded-full bg-base-content/50"></span>
							Checking...
						</div>
					{/if}
				</div>
			</div>

			<!-- Search Controls -->
			<div class="grid grid-cols-1 md:grid-cols-2 gap-4">
				{#if locationsWithCoords.length > 0}
					<div class="form-control">
						<label class="label">
							<span class="label-text font-semibold">📍 Search around</span>
						</label>
						<select class="select select-bordered w-full" bind:value={selectedLocationId}>
							{#each locationsWithCoords as location}
								<option value={location.id}>{location.name}</option>
							{/each}
						</select>
					</div>
				{:else}
					<div class="alert alert-warning">
						<span>No locations with coordinates in this collection. Add locations with coordinates first.</span>
					</div>
				{/if}

				<div class="form-control">
					<label class="label">
						<span class="label-text font-semibold">🎯 Search radius: {radiusDisplay}</span>
					</label>
					<input
						type="range"
						min="1"
						max="50"
						step="1"
						class="range range-primary range-sm"
						bind:value={radiusKm}
					/>
					<div class="flex justify-between text-xs opacity-60 px-1">
						<span>{isMetric ? '1 km' : '0.6 mi'}</span>
						<span>{isMetric ? '50 km' : '31 mi'}</span>
					</div>
				</div>
			</div>

			<!-- Action Buttons -->
			<div class="flex gap-2 mt-4">
				<button
					class="btn btn-primary flex-1"
					on:click={searchGemSpot}
					disabled={loading || !selectedLocationId}
				>
					{#if loading}
						<span class="loading loading-spinner loading-sm"></span>
						Analyzing...
					{:else}
						<RobotOutline class="w-5 h-5" />
						Find Hidden Gems
					{/if}
				</button>
				<button class="btn btn-ghost" on:click={() => (showFilters = !showFilters)}>
					<TuneVariant class="w-5 h-5" />
				</button>
			</div>

			<!-- Filters -->
			{#if showFilters}
				<div class="divider">Filters</div>
				<div class="grid grid-cols-1 md:grid-cols-2 gap-4">
					<div class="form-control">
						<label class="label">
							<span class="label-text">Max results</span>
						</label>
						<select class="select select-bordered select-sm" bind:value={topK}>
							<option value={5}>Top 5</option>
							<option value={10}>Top 10</option>
							<option value={20}>Top 20</option>
							<option value={50}>Top 50</option>
						</select>
					</div>
				</div>
			{/if}

			<!-- Error -->
			{#if error}
				<div class="alert alert-error mt-4">
					<CloseCircle class="w-6 h-6" />
					<span>{error}</span>
				</div>
			{/if}
		</div>
	</div>

	<!-- Results -->
	{#if loading}
		<div class="flex flex-col items-center justify-center py-16 gap-4">
			<span class="loading loading-spinner loading-lg text-primary"></span>
			<p class="text-sm opacity-70 animate-pulse">Running XGBoost inference...</p>
		</div>
	{:else if gemCards.length > 0}
		<!-- Stats Bar -->
		<div class="stats shadow w-full bg-base-200">
			<div class="stat">
				<div class="stat-title">Gem Cards</div>
				<div class="stat-value text-primary">{gemCards.length}</div>
			</div>
			<div class="stat">
				<div class="stat-title">Candidates Scored</div>
				<div class="stat-value text-secondary">{candidatesScored}</div>
			</div>
			<div class="stat">
				<div class="stat-title">Inference Time</div>
				<div class="stat-value text-accent">{inferenceTimeMs.toFixed(0)}ms</div>
			</div>
			<div class="stat">
				<div class="stat-title">Model</div>
				<div class="stat-value text-sm">{modelVersion}</div>
			</div>
		</div>

		<!-- Map View -->
		<div class="card bg-base-200 shadow-xl">
			<div class="card-body">
				<h3 class="card-title text-xl mb-4">
					<RobotOutline class="w-6 h-6" />
					AI Recommendations Map
				</h3>
				<div class="rounded-lg overflow-hidden shadow-lg">
					<MapLibre
						style={getBasemapUrl()}
						class="w-full h-[500px]"
						standardControls
						center={mapCenter}
						zoom={mapZoom}
					>
						<!-- Collection Locations (blue markers) -->
						{#each collection.locations as location}
							{#if location.latitude && location.longitude}
								<DefaultMarker lngLat={{ lng: location.longitude, lat: location.latitude }}>
									<Popup openOn="click" offset={[0, -10]}>
										<div class="p-2">
											<a
												href={`/adventures/${location.id}`}
												class="text-lg font-bold text-black hover:underline mb-1 block"
											>
												{location.name}
											</a>
											<p class="text-xs text-black opacity-70">Your location</p>
										</div>
									</Popup>
								</DefaultMarker>
							{/if}
						{/each}

						<!-- Gem Card Results -->
						{#each gemCards as card}
							<DefaultMarker lngLat={{ lng: card.longitude, lat: card.latitude }}>
								<Popup openOn="click" offset={[0, -10]}>
									<div class="p-3 max-w-xs">
										<h4 class="text-base font-bold text-black mb-1">{card.name}</h4>
										<p class="text-xs text-black font-semibold mb-1">
											Score: {card.confidence_pct}
										</p>
										<div class="flex gap-1 flex-wrap mb-2">
											{#each card.vibe_tags as tag}
												<span class="text-xs px-1.5 py-0.5 rounded bg-primary/20 text-black"
													>{tag}</span
												>
											{/each}
										</div>
										<p class="text-xs text-black">
											🚶 {formatDistance(card.distance_km)} away
										</p>
									</div>
								</Popup>
							</DefaultMarker>
						{/each}
					</MapLibre>
				</div>
			</div>
		</div>

		<!-- Gem Cards Grid -->
		<div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
			{#each gemCards as card, i}
				<div
					class="card bg-base-100 shadow-xl hover:shadow-2xl transition-all duration-300 border border-base-300 hover:border-primary/30"
				>
					<!-- Gradient Top Bar -->
					<div
						class="h-2 rounded-t-2xl"
						style="background: linear-gradient(90deg, hsl(var(--p)) {card.score * 100}%, hsl(var(--b3)) {card.score * 100}%)"
					></div>

					<div class="card-body p-4">
						<!-- Rank & Title -->
						<div class="flex items-start gap-3">
							<div
								class="w-10 h-10 rounded-full bg-gradient-to-br from-primary to-secondary flex items-center justify-center text-primary-content font-bold text-sm shrink-0 shadow-md"
							>
								#{i + 1}
							</div>
							<div class="flex-1 min-w-0">
								<h3 class="card-title text-lg leading-tight">{card.name}</h3>
								<p class="text-xs opacity-60 mt-0.5">{card.category}</p>
							</div>
						</div>

						<!-- Confidence Score -->
						<div class="flex items-center gap-2 mt-2">
							<div class="text-sm font-semibold">AI Confidence:</div>
							<div class="badge {getScoreColor(card.score)} badge-lg font-bold">
								{card.confidence_pct}
							</div>
						</div>

						<!-- Vibe Tags -->
						{#if card.vibe_tags && card.vibe_tags.length > 0}
							<div class="flex gap-1.5 flex-wrap mt-2">
								{#each card.vibe_tags as tag}
									<span
										class="text-xs px-2.5 py-1 rounded-full border {getVibeTagColor(tag)} font-medium"
									>
										{tag}
									</span>
								{/each}
							</div>
						{/if}

						<!-- Rating & Distance -->
						<div class="flex gap-2 flex-wrap mt-3">
							{#if card.avg_rating > 0}
								<div class="flex items-center gap-1">
									<div class="flex text-yellow-500">
										{#each renderStars(card.avg_rating) as star}
											{#if star.type === 'full'}
												<Star class="w-3.5 h-3.5" />
											{:else if star.type === 'half'}
												<StarHalfFull class="w-3.5 h-3.5" />
											{:else}
												<StarOutline class="w-3.5 h-3.5" />
											{/if}
										{/each}
									</div>
									<span class="text-xs font-semibold">{card.avg_rating.toFixed(1)}</span>
								</div>
							{/if}
							<div class="badge badge-outline badge-sm">
								🚶 {formatDistance(card.distance_km)}
							</div>
						</div>

						<!-- Actions -->
						<div class="card-actions justify-end mt-4">
							<button
								class="btn btn-sm btn-primary"
								on:click={() => openCreateLocationFromGemCard(card)}
							>
								+ Add to Collection
							</button>
						</div>
					</div>
				</div>
			{/each}
		</div>
	{:else if !loading && gemCards.length === 0 && !error}
		<div class="card bg-base-200 shadow-xl">
			<div class="card-body text-center py-16">
				<RobotOutline class="w-24 h-24 mx-auto opacity-20 mb-4" />
				<h3 class="text-2xl font-bold mb-2">Ready to Discover Hidden Gems</h3>
				<p class="opacity-70 max-w-md mx-auto">
					Select a location from your collection and click "Find Hidden Gems" to get
					AI-powered recommendations scored by our XGBoost model.
				</p>
			</div>
		</div>
	{/if}
</div>
