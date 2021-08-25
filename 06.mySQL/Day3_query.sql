SELECT `name`, debut, title
 FROM girl_group
 JOIN song
 ON girl_group.hit_song_id=song.sid
 WHERE debut between '2009-01-01' and '2009-12-31'
 ORDER BY debut;
 