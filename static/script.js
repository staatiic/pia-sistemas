$(document).ready(function() {
    const movies = window.moviesData;
    const albums = window.albumsData;
    let selectedMovie = null;
    let selectedAlbum = null;

    window.showMovieSearch = function() {
        $('#movieSection').show();
        $('#musicSection').hide();
    };

    window.showMusicSearch = function() {
        $('#musicSection').show();
        $('#movieSection').hide();
    };

    function searchMovies(query) {
        if (!query) {
            $('#movieSearchResults').hide();
            return;
        }

        const results = movies.filter(movie => 
            movie.title.toLowerCase().includes(query.toLowerCase())
        );

        const searchResults = $('#movieSearchResults');
        searchResults.empty();
        
        if (results.length > 0) {
            results.forEach(movie => {
                const div = $('<div>')
                    .addClass('search-item')
                    .html(`
                        <div class="d-flex justify-content-between align-items-center">
                            <div>
                                <strong>${movie.title}</strong>
                                <div class="text-muted small">${movie.genres}</div>
                            </div>
                            <div class="text-warning">
                                <i class="fas fa-star"></i> ${movie.vote_average}
                            </div>
                        </div>
                    `)
                    .click(() => selectMovie(movie.title));
                searchResults.append(div);
            });
            searchResults.show();
        } else {
            searchResults.hide();
        }
    }

    function searchAlbums(query) {
        if (!query) {
            $('#albumSearchResults').hide();
            return;
        }

        const results = albums.filter(album => 
            album.title.toLowerCase().includes(query.toLowerCase()) ||
            album.artist.toLowerCase().includes(query.toLowerCase())
        );

        const searchResults = $('#albumSearchResults');
        searchResults.empty();
        
        if (results.length > 0) {
            results.forEach(album => {
                const div = $('<div>')
                    .addClass('search-item')
                    .html(`
                        <div class="d-flex justify-content-between align-items-center">
                            <div>
                                <strong>${album.title}</strong>
                                <div class="text-muted small">${album.artist}</div>
                            </div>
                            <div class="text-warning">
                                <i class="fas fa-star"></i> ${album.rating}
                            </div>
                        </div>
                    `)
                    .click(() => selectAlbum(album.title));
                searchResults.append(div);
            });
            searchResults.show();
        } else {
            searchResults.hide();
        }
    }

    function selectMovie(movieTitle) {
        selectedMovie = movieTitle;
        $('#movieSearch').val(movieTitle);
        $('#movieSearchResults').hide();
        getMovieRecommendations(movieTitle);
    }

    function selectAlbum(albumTitle) {
        selectedAlbum = albumTitle;
        $('#albumSearch').val(albumTitle);
        $('#albumSearchResults').hide();
        getAlbumRecommendations(albumTitle);
    }

    function getMovieRecommendations(movieTitle) {
        if (!movieTitle) return;

        $.ajax({
            url: '/get_recommendations',
            method: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({ movie: movieTitle }),
            success: function(response) {
                const container = $('#recommendationsContainer');
                container.empty();
                response.forEach(function(movie) {
                    const movieCard = `
                        <div class="movie-card">
                            <img src="${movie.poster_path || 'https://via.placeholder.com/300x450?text=No+Image'}" 
                                 class="movie-poster" 
                                 alt="${movie.title}">
                            <div class="card-body">
                                <h5 class="card-title">${movie.title}</h5>
                                <p class="card-text">Géneros: ${movie.genres}</p>
                                <p class="card-text">Rating: ${movie.vote_average.toFixed(1)} ⭐</p>
                            </div>
                        </div>
                    `;
                    container.append(movieCard);
                });
                $('#movieRecommendations').show();
            },
            error: function(xhr, status, error) {
                console.error('Error:', error);
                alert('Error al obtener recomendaciones');
            }
        });
    }

    function getAlbumRecommendations(albumTitle) {
        if (!albumTitle) return;

        $.ajax({
            url: '/get_music_recommendations',
            method: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({ album: albumTitle }),
            success: function(response) {
                const container = $('#musicRecommendationsContainer');
                container.empty();
                response.forEach(function(album) {
                    const albumCard = `
                        <div class="album-card">
                            <img src="${album.image_url || 'https://via.placeholder.com/300x300?text=No+Image'}" 
                                 class="album-cover" 
                                 alt="${album.title}">
                            <div class="card-body">
                                <h5 class="card-title">${album.title}</h5>
                                <p class="card-text"><strong>${album.artist}</strong></p>
                                <p class="card-text">Géneros: ${album.genres}</p>
                                <p class="card-text">Rating: ${album.rating.toFixed(1)} ⭐</p>
                            </div>
                        </div>
                    `;
                    container.append(albumCard);
                });
                $('#musicRecommendations').show();
            },
            error: function(xhr, status, error) {
                console.error('Error:', error);
                alert('Error al obtener recomendaciones de música');
            }
        });
    }

    $('#movieSearch').on('input', function() {
        searchMovies($(this).val());
    });

    $('#albumSearch').on('input', function() {
        searchAlbums($(this).val());
    });

    $(document).on('click', function(e) {
        if (!$(e.target).closest('.search-container').length) {
            $('.search-results').hide();
        }
    });
    showMovieSearch();
});