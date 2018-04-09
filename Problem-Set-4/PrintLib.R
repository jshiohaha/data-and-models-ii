psr = function(s,r,d) {
  rr = format(round(r, d), nsmall=d, big.mark=",", scientific=FALSE)
  print (paste(s,as.character(rr),sep=' '))
}

psi = function(s,i) {
  print (paste(s,as.character(i),sep=' '))
}

psir = function(s,i) {
  ii = format(i, big.mark=",", scientific=FALSE)
  return (paste(s,as.character(ii),sep=' '))
}

ps2 = function(s1,i,s2,r) {
  t = paste(s1, as.character(i),sep=' ')
  t = paste(t, s2,sep=' ')
  return (paste(t, as.character(round(r*100,3)),sep=' ') ) 
}

ps3 = function(s1,i,s2,j,s3,r) {
  t1 = paste(s1,as.character(i),sep=' ')
  t2 = paste(s2,as.character(j),sep=' ')
  t = paste(t1,t2,sep=' ')
  t3 = paste(s3,as.character(round(r*100,3)),sep=' ')
  t = paste(t,t3)
  return (paste(t,"%"))
}

psii = function(s, v) {
  t = paste(s,as.character(v[1]),sep=' ')
  return (  paste(t,as.character(v[2]),sep=' ') ) 
}

psv = function(s,v,d) {
  t1 = paste(s,as.character(round(v[1],d)),sep=' ')
  t2 = paste(t1,as.character(round(v[2],d)),sep=' ')
  return (paste(t2))
}

psp = function(s,r,d) {
  print (paste(s,as.character(round(r*100,d)),"%",sep=' '))
}

psvi = function(s,u) {
	c = paste(s,as.character(u[1]),sep=' ')
	v = sort(u)
	n = length(v)
	if (n > 1) {
	for (i in 2:n) {
		c = paste(c,as.character(v[i]),sep=' ')
	}
}
  print (c)
}

psrr = function(s,r,d) {
  rr = format(round(r, d), nsmall=d, big.mark=",", scientific=FALSE)
  return (paste(s,as.character(rr),sep=' '))
}

psir = function(s,i) {
  return (paste(s,as.character(i),sep=' '))
}

ps2r = function(s1,i,s2,r) {
  t = paste(s1, as.character(i),sep=' ')
  t = paste(t, s2,sep=' ')
  return (paste(t, as.character(round(r*100,3)),sep=' ') ) 
}

pspr = function(s,r,d) {
  return (paste(s,as.character(round(r*100,d)),"%",sep=' '))
}

pSet = function(set,vect) {
  c = paste('Set',set, sep= ' ')
  c = paste(c,':  ', sep = ' ')
  v = sort(vect)
	for (i in 1:length(v) ) {
		c = paste(c,as.character(v[i]),sep=' ')
}
  print (c)
}