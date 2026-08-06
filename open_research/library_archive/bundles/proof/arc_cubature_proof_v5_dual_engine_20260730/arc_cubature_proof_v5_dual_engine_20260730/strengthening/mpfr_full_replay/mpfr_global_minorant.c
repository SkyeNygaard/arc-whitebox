#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <gmp.h>
/* Minimal MPFR ABI declarations; the runtime library is present but development headers are not. */
typedef long int mpfr_prec_t;
typedef long int mpfr_exp_t;
typedef int mpfr_sign_t;
typedef enum { MPFR_RNDN=0, MPFR_RNDZ=1, MPFR_RNDU=2, MPFR_RNDD=3, MPFR_RNDA=4, MPFR_RNDF=5, MPFR_RNDNA=-1 } mpfr_rnd_t;
typedef struct { mpfr_prec_t _mpfr_prec; mpfr_sign_t _mpfr_sign; mpfr_exp_t _mpfr_exp; mp_limb_t *_mpfr_d; } __mpfr_struct;
typedef __mpfr_struct mpfr_t[1];
typedef __mpfr_struct *mpfr_ptr;
typedef const __mpfr_struct *mpfr_srcptr;
void mpfr_set_default_prec(mpfr_prec_t);
void mpfr_init2(mpfr_ptr, mpfr_prec_t);
void mpfr_clear(mpfr_ptr);
int mpfr_set(mpfr_ptr, mpfr_srcptr, mpfr_rnd_t);
int mpfr_set_ui(mpfr_ptr, unsigned long, mpfr_rnd_t);
int mpfr_set_q(mpfr_ptr, mpq_srcptr, mpfr_rnd_t);
int mpfr_add(mpfr_ptr, mpfr_srcptr, mpfr_srcptr, mpfr_rnd_t);
int mpfr_sub(mpfr_ptr, mpfr_srcptr, mpfr_srcptr, mpfr_rnd_t);
int mpfr_mul(mpfr_ptr, mpfr_srcptr, mpfr_srcptr, mpfr_rnd_t);
int mpfr_div(mpfr_ptr, mpfr_srcptr, mpfr_srcptr, mpfr_rnd_t);
int mpfr_div_ui(mpfr_ptr, mpfr_srcptr, unsigned long, mpfr_rnd_t);
int mpfr_ui_div(mpfr_ptr, unsigned long, mpfr_srcptr, mpfr_rnd_t);
int mpfr_mul_ui(mpfr_ptr, mpfr_srcptr, unsigned long, mpfr_rnd_t);
int mpfr_neg(mpfr_ptr, mpfr_srcptr, mpfr_rnd_t);
int mpfr_abs(mpfr_ptr, mpfr_srcptr, mpfr_rnd_t);
int mpfr_sqrt(mpfr_ptr, mpfr_srcptr, mpfr_rnd_t);
int mpfr_asin(mpfr_ptr, mpfr_srcptr, mpfr_rnd_t);
int mpfr_const_pi(mpfr_ptr, mpfr_rnd_t);
int mpfr_cmp(mpfr_srcptr, mpfr_srcptr);
int mpfr_cmp_si(mpfr_srcptr, long);
int mpfr_cmp_ui(mpfr_srcptr, unsigned long);
int mpfr_sgn(mpfr_srcptr);
void mpfr_set_inf(mpfr_ptr, int);
int __gmpfr_out_str(FILE*, int, size_t, mpfr_srcptr, mpfr_rnd_t);
#define mpfr_out_str __gmpfr_out_str
int mpfr_asprintf(char**, const char*, ...);
void mpfr_free_str(char*);
const char *mpfr_get_version(void);

#define PREC 256
#define MAX_LINE 8192
#define MAX_DEPTH 12

typedef struct { mpfr_t lo, hi; } I;
static void ii(I *x){ mpfr_init2(x->lo,PREC); mpfr_init2(x->hi,PREC); }
static void ci(I *x){ mpfr_clear(x->lo); mpfr_clear(x->hi); }
static void set_ui_i(I *x,unsigned long n){mpfr_set_ui(x->lo,n,MPFR_RNDD);mpfr_set_ui(x->hi,n,MPFR_RNDU);} 
static void set_q_i(I *x,const mpq_t q){mpfr_set_q(x->lo,q,MPFR_RNDD);mpfr_set_q(x->hi,q,MPFR_RNDU);} 
static void copy_i(I *r,const I *a){mpfr_set(r->lo,a->lo,MPFR_RNDD);mpfr_set(r->hi,a->hi,MPFR_RNDU);} 
static void add_i(I *r,const I*a,const I*b){mpfr_add(r->lo,a->lo,b->lo,MPFR_RNDD);mpfr_add(r->hi,a->hi,b->hi,MPFR_RNDU);} 
static void neg_i(I*r,const I*a){mpfr_neg(r->lo,a->hi,MPFR_RNDD);mpfr_neg(r->hi,a->lo,MPFR_RNDU);} 
static void sub_i(I*r,const I*a,const I*b){I nb;ii(&nb);neg_i(&nb,b);add_i(r,a,&nb);ci(&nb);} 
static void mul_i(I*r,const I*a,const I*b){
 mpfr_t dl[4],du[4]; for(int k=0;k<4;k++){mpfr_init2(dl[k],PREC);mpfr_init2(du[k],PREC);} 
 mpfr_mul(dl[0],a->lo,b->lo,MPFR_RNDD); mpfr_mul(du[0],a->lo,b->lo,MPFR_RNDU);
 mpfr_mul(dl[1],a->lo,b->hi,MPFR_RNDD); mpfr_mul(du[1],a->lo,b->hi,MPFR_RNDU);
 mpfr_mul(dl[2],a->hi,b->lo,MPFR_RNDD); mpfr_mul(du[2],a->hi,b->lo,MPFR_RNDU);
 mpfr_mul(dl[3],a->hi,b->hi,MPFR_RNDD); mpfr_mul(du[3],a->hi,b->hi,MPFR_RNDU);
 mpfr_set(r->lo,dl[0],MPFR_RNDD);mpfr_set(r->hi,du[0],MPFR_RNDU);
 for(int k=1;k<4;k++){if(mpfr_cmp(dl[k],r->lo)<0)mpfr_set(r->lo,dl[k],MPFR_RNDD);if(mpfr_cmp(du[k],r->hi)>0)mpfr_set(r->hi,du[k],MPFR_RNDU);} 
 for(int k=0;k<4;k++){mpfr_clear(dl[k]);mpfr_clear(du[k]);}
}
static void div_i(I*r,const I*a,const I*b){
 if(mpfr_sgn(b->lo)<=0 && mpfr_sgn(b->hi)>=0){fprintf(stderr,"division by interval containing zero\n");exit(3);} 
 I inv;ii(&inv);mpfr_ui_div(inv.lo,1,b->hi,MPFR_RNDD);mpfr_ui_div(inv.hi,1,b->lo,MPFR_RNDU);mul_i(r,a,&inv);ci(&inv);
}
static void sqrt_i(I*r,const I*a){if(mpfr_sgn(a->lo)<0){fprintf(stderr,"sqrt negative\n");exit(4);}mpfr_sqrt(r->lo,a->lo,MPFR_RNDD);mpfr_sqrt(r->hi,a->hi,MPFR_RNDU);} 
static void asin_i(I*r,const I*a){mpfr_asin(r->lo,a->lo,MPFR_RNDD);mpfr_asin(r->hi,a->hi,MPFR_RNDU);} 
static void abs_mpfr(mpfr_t r,const mpfr_t x,mpfr_rnd_t rnd){mpfr_abs(r,x,rnd);} 

static I PI;

static void kappa_point(I *k,I *kp,const mpfr_t x){
 if(mpfr_cmp_si(x,-1)==0){set_ui_i(k,0);set_ui_i(kp,0);return;} if(mpfr_cmp_ui(x,1)==0){set_ui_i(k,1);set_ui_i(kp,1);return;}
 I X,one,x2,omx2,root,ax,half,pi2,angle,prod,num;ii(&X);ii(&one);ii(&x2);ii(&omx2);ii(&root);ii(&ax);ii(&half);ii(&pi2);ii(&angle);ii(&prod);ii(&num);
 mpfr_set(X.lo,x,MPFR_RNDD);mpfr_set(X.hi,x,MPFR_RNDU);set_ui_i(&one,1);
 mul_i(&x2,&X,&X);sub_i(&omx2,&one,&x2);sqrt_i(&root,&omx2);asin_i(&ax,&X);
 mpfr_set_ui(half.lo,1,MPFR_RNDD);mpfr_div_ui(half.lo,half.lo,2,MPFR_RNDD);mpfr_set_ui(half.hi,1,MPFR_RNDU);mpfr_div_ui(half.hi,half.hi,2,MPFR_RNDU);
 // pi/2 computed directly
 mpfr_div_ui(pi2.lo,PI.lo,2,MPFR_RNDD);mpfr_div_ui(pi2.hi,PI.hi,2,MPFR_RNDU);
 add_i(&angle,&pi2,&ax);mul_i(&prod,&angle,&X);add_i(&num,&root,&prod);div_i(k,&num,&PI);div_i(&prod,&ax,&PI);add_i(kp,&half,&prod);
 ci(&X);ci(&one);ci(&x2);ci(&omx2);ci(&root);ci(&ax);ci(&half);ci(&pi2);ci(&angle);ci(&prod);ci(&num);
}

static void kappa_interval(I *k,I *kp,const I *x){
 I kl,kpl,kh,kph;ii(&kl);ii(&kpl);ii(&kh);ii(&kph);kappa_point(&kl,&kpl,x->lo);kappa_point(&kh,&kph,x->hi);
 mpfr_set(k->lo,kl.lo,MPFR_RNDD);mpfr_set(k->hi,kh.hi,MPFR_RNDU);mpfr_set(kp->lo,kpl.lo,MPFR_RNDD);mpfr_set(kp->hi,kph.hi,MPFR_RNDU);
 ci(&kl);ci(&kpl);ci(&kh);ci(&kph);
}

static void kappa_second(I *r,const I*x){
 mpfr_t alo,ahi,maxabs,minabs;mpfr_init2(alo,PREC);mpfr_init2(ahi,PREC);mpfr_init2(maxabs,PREC);mpfr_init2(minabs,PREC);
 abs_mpfr(alo,x->lo,MPFR_RNDU);abs_mpfr(ahi,x->hi,MPFR_RNDU);if(mpfr_cmp(alo,ahi)>=0)mpfr_set(maxabs,alo,MPFR_RNDU);else mpfr_set(maxabs,ahi,MPFR_RNDU);
 if(mpfr_sgn(x->lo)<=0 && mpfr_sgn(x->hi)>=0)mpfr_set_ui(minabs,0,MPFR_RNDD);else {if(mpfr_cmp(alo,ahi)<=0)mpfr_set(minabs,alo,MPFR_RNDD);else mpfr_set(minabs,ahi,MPFR_RNDD);} 
 I one,minA,maxA,minSq,maxSq,minArg,maxArg,smin,smax,den;ii(&one);ii(&minA);ii(&maxA);ii(&minSq);ii(&maxSq);ii(&minArg);ii(&maxArg);ii(&smin);ii(&smax);ii(&den);set_ui_i(&one,1);
 mpfr_set(minA.lo,minabs,MPFR_RNDD);mpfr_set(minA.hi,minabs,MPFR_RNDU);mpfr_set(maxA.lo,maxabs,MPFR_RNDD);mpfr_set(maxA.hi,maxabs,MPFR_RNDU);
 mul_i(&minSq,&minA,&minA);mul_i(&maxSq,&maxA,&maxA);sub_i(&minArg,&one,&maxSq);sub_i(&maxArg,&one,&minSq);sqrt_i(&smin,&minArg);sqrt_i(&smax,&maxArg);
 mpfr_mul(den.lo,PI.lo,smin.lo,MPFR_RNDD);mpfr_mul(den.hi,PI.hi,smax.hi,MPFR_RNDU);I oneI;ii(&oneI);set_ui_i(&oneI,1);div_i(r,&oneI,&den);ci(&oneI);
 ci(&one);ci(&minA);ci(&maxA);ci(&minSq);ci(&maxSq);ci(&minArg);ci(&maxArg);ci(&smin);ci(&smax);ci(&den);mpfr_clear(alo);mpfr_clear(ahi);mpfr_clear(maxabs);mpfr_clear(minabs);
}

static void deep_second(I*r,const mpq_t aq,const mpq_t bq){
 I z,p,q;ii(&z);ii(&p);ii(&q);mpfr_set_q(z.lo,aq,MPFR_RNDD);mpfr_set_q(z.hi,bq,MPFR_RNDU);set_ui_i(&p,1);set_ui_i(&q,0);
 for(int it=0;it<32;it++){
  I kp2,znew,kp,p2,t1,t2,qnew,pnew;ii(&kp2);ii(&znew);ii(&kp);ii(&p2);ii(&t1);ii(&t2);ii(&qnew);ii(&pnew);
  kappa_second(&kp2,&z);kappa_interval(&znew,&kp,&z);mul_i(&p2,&p,&p);mul_i(&t1,&kp2,&p2);mul_i(&t2,&kp,&q);add_i(&qnew,&t1,&t2);mul_i(&pnew,&kp,&p);
  copy_i(&z,&znew);copy_i(&p,&pnew);copy_i(&q,&qnew);
  ci(&kp2);ci(&znew);ci(&kp);ci(&p2);ci(&t1);ci(&t2);ci(&qnew);ci(&pnew);
 }
 copy_i(r,&q);ci(&z);ci(&p);ci(&q);
}

static mpq_t HPPQ[4];
static I HPPI[4];
static void hpp_range(I*r,const mpq_t aq,const mpq_t bq){
 I X,y,tmp;ii(&X);ii(&y);ii(&tmp);mpfr_set_q(X.lo,aq,MPFR_RNDD);mpfr_set_q(X.hi,bq,MPFR_RNDU);set_ui_i(&y,0);
 for(int k=3;k>=0;k--){mul_i(&tmp,&y,&X);add_i(&y,&tmp,&HPPI[k]);}
 copy_i(r,&y);ci(&X);ci(&y);ci(&tmp);
}
static void gpp_range(I*r,const mpq_t a,const mpq_t b){I h,k;ii(&h);ii(&k);hpp_range(&h,a,b);deep_second(&k,a,b);sub_i(r,&h,&k);ci(&h);ci(&k);} 

static int parse_q(mpq_t q,const char*s){if(mpq_set_str(q,s,10)!=0)return 0;mpq_canonicalize(q);return 1;}

typedef struct {long accepted;long split;int maxdepth;mpfr_t minmargin;} Stats;
static void stats_init(Stats*s){s->accepted=0;s->split=0;s->maxdepth=0;mpfr_init2(s->minmargin,PREC);mpfr_set_inf(s->minmargin,1);} 
static void recurse(const mpq_t a,const mpq_t b,int sign,int depth,Stats*s){
 I R;ii(&R);gpp_range(&R,a,b);int good=sign<0?mpfr_sgn(R.hi)<0:mpfr_sgn(R.lo)>0;
 if(good){s->accepted++;if(depth>s->maxdepth)s->maxdepth=depth;mpfr_t m;mpfr_init2(m,PREC);if(sign<0)mpfr_neg(m,R.hi,MPFR_RNDD);else mpfr_set(m,R.lo,MPFR_RNDD);if(mpfr_cmp(m,s->minmargin)<0)mpfr_set(s->minmargin,m,MPFR_RNDD);mpfr_clear(m);ci(&R);return;}
 if(depth>=MAX_DEPTH){fprintf(stderr,"FAILED depth %d: ",depth);mpq_out_str(stderr,10,a);fprintf(stderr," ");mpq_out_str(stderr,10,b);fprintf(stderr," range=");mpfr_out_str(stderr,10,0,R.lo,MPFR_RNDD);fprintf(stderr,",");mpfr_out_str(stderr,10,0,R.hi,MPFR_RNDU);fprintf(stderr,"\n");ci(&R);exit(5);} 
 s->split++;mpq_t m;mpq_init(m);mpq_add(m,a,b);mpq_div_2exp(m,m,1);recurse(a,m,sign,depth+1,s);recurse(m,b,sign,depth+1,s);mpq_clear(m);ci(&R);
}

static void load_hpp(const char*path){FILE*f=fopen(path,"r");if(!f){perror(path);exit(2);}char line[MAX_LINE];for(int i=0;i<4;i++){if(!fgets(line,sizeof line,f)){fprintf(stderr,"hpp missing\n");exit(2);}line[strcspn(line,"\r\n")]=0;mpq_init(HPPQ[i]);if(!parse_q(HPPQ[i],line)){fprintf(stderr,"bad hpp\n");exit(2);}ii(&HPPI[i]);set_q_i(&HPPI[i],HPPQ[i]);}fclose(f);} 

static void write_json(const char*out,const char*mode,long initial,long accepted,long split,int maxdepth,const mpfr_t margin,int passed){FILE*f=fopen(out,"w");if(!f){perror(out);exit(2);}char*ms=NULL;mpfr_asprintf(&ms,"%.70Re",margin);fprintf(f,"{\n  \"engine\": \"MPFR %s, direct C, 256-bit precision\",\n  \"mode\": \"%s\",\n  \"initial_intervals\": %ld,\n  \"accepted_intervals\": %ld,\n  \"splits\": %ld,\n  \"maximum_depth\": %d,\n  \"minimum_sign_margin\": \"%s\",\n  \"passed\": %s\n}\n",mpfr_get_version(),mode,initial,accepted,split,maxdepth,ms,passed?"true":"false");mpfr_free_str(ms);fclose(f);} 


static mpq_t HQ[6], HPQ[5];
static I HI[6], HPI[5];
static void load_poly(const char*path, mpq_t *qs, I *is, int n){FILE*f=fopen(path,"r");if(!f){perror(path);exit(2);}char line[MAX_LINE];for(int i=0;i<n;i++){if(!fgets(line,sizeof line,f)){fprintf(stderr,"poly missing\n");exit(2);}line[strcspn(line,"\r\n")]=0;mpq_init(qs[i]);if(!parse_q(qs[i],line)){fprintf(stderr,"bad poly\n");exit(2);}ii(&is[i]);set_q_i(&is[i],qs[i]);}fclose(f);}
static void poly_range(I*r,const I *coef,int n,const mpq_t aq,const mpq_t bq){I X,y,tmp;ii(&X);ii(&y);ii(&tmp);mpfr_set_q(X.lo,aq,MPFR_RNDD);mpfr_set_q(X.hi,bq,MPFR_RNDU);set_ui_i(&y,0);for(int k=n-1;k>=0;k--){mul_i(&tmp,&y,&X);add_i(&y,&tmp,&coef[k]);}copy_i(r,&y);ci(&X);ci(&y);ci(&tmp);}
static void deep_pair(I*k,I*p,const mpq_t aq,const mpq_t bq){I z,prod;ii(&z);ii(&prod);mpfr_set_q(z.lo,aq,MPFR_RNDD);mpfr_set_q(z.hi,bq,MPFR_RNDU);set_ui_i(&prod,1);for(int it=0;it<32;it++){I zn,kp,pn;ii(&zn);ii(&kp);ii(&pn);kappa_interval(&zn,&kp,&z);mul_i(&pn,&prod,&kp);copy_i(&z,&zn);copy_i(&prod,&pn);ci(&zn);ci(&kp);ci(&pn);}copy_i(k,&z);copy_i(p,&prod);ci(&z);ci(&prod);}
static void gp_range2(I*r,const mpq_t a,const mpq_t b){I hp,k,p;ii(&hp);ii(&k);ii(&p);poly_range(&hp,HPI,5,a,b);deep_pair(&k,&p,a,b);sub_i(r,&hp,&p);ci(&hp);ci(&k);ci(&p);}
static void poly_q_eval(mpq_t out, mpq_t *coef, int n, const mpq_t x){mpq_set_ui(out,0,1);for(int k=n-1;k>=0;k--){mpq_mul(out,out,x);mpq_add(out,out,coef[k]);}}
static void gp_point2(I*r,const mpq_t x){mpq_t hx;mpq_init(hx);poly_q_eval(hx,HPQ,5,x);I h,k,p;ii(&h);ii(&k);ii(&p);set_q_i(&h,hx);deep_pair(&k,&p,x,x);sub_i(r,&h,&p);ci(&h);ci(&k);ci(&p);mpq_clear(hx);}
static void g_box_upper(mpfr_t out,const mpq_t a,const mpq_t b){mpq_t hb;mpq_init(hb);poly_q_eval(hb,HQ,6,b);I h,k,p;ii(&h);ii(&k);ii(&p);set_q_i(&h,hb);deep_pair(&k,&p,a,a);mpfr_sub(out,h.hi,k.lo,MPFR_RNDU);ci(&h);ci(&k);ci(&p);mpq_clear(hb);}
static void g_point2(I*r,const mpq_t x){mpq_t hx;mpq_init(hx);poly_q_eval(hx,HQ,6,x);I h,k,p;ii(&h);ii(&k);ii(&p);set_q_i(&h,hx);deep_pair(&k,&p,x,x);sub_i(r,&h,&k);ci(&h);ci(&k);ci(&p);mpq_clear(hx);}
static char* mstr(const mpfr_t x){char*s=NULL;mpfr_asprintf(&s,"%.70Re",x);return s;}

int main(int argc,char**argv){
 if(argc!=7){fprintf(stderr,"usage: %s h.tsv hp.tsv hpp.tsv global.tsv output.json log.txt\n",argv[0]);return 2;}
 mpfr_set_default_prec(PREC);ii(&PI);mpfr_const_pi(PI.lo,MPFR_RNDD);mpfr_const_pi(PI.hi,MPFR_RNDU);
 load_poly(argv[1],HQ,HI,6);load_poly(argv[2],HPQ,HPI,5);load_hpp(argv[3]);
 FILE*f=fopen(argv[4],"r"),*log=fopen(argv[6],"w");if(!f||!log){perror("open");return 2;}
 char line[MAX_LINE];int rows=0,crit=0,infl=0,endc=0,tail=0;mpfr_t global_ub,min_gp_margin,min_gpp_margin;mpfr_init2(global_ub,PREC);mpfr_init2(min_gp_margin,PREC);mpfr_init2(min_gpp_margin,PREC);mpfr_set_inf(global_ub,-1);mpfr_set_inf(min_gp_margin,1);mpfr_set_inf(min_gpp_margin,1);
 while(fgets(line,sizeof line,f)){char*save=NULL;char*kind=strtok_r(line,"\t\r\n",&save);char*name=strtok_r(NULL,"\t\r\n",&save);char*as=strtok_r(NULL,"\t\r\n",&save);char*bs=strtok_r(NULL,"\t\r\n",&save);char*sg=strtok_r(NULL,"\t\r\n",&save);char*cs=strtok_r(NULL,"\t\r\n",&save);char*rs=strtok_r(NULL,"\t\r\n",&save);if(!kind||!name||!as||!bs||!sg){fprintf(stderr,"bad global row\n");return 2;}mpq_t a,b;mpq_init(a);mpq_init(b);parse_q(a,as);parse_q(b,bs);rows++;
  if(strcmp(kind,"CRIT")==0){crit++;I R,gl,gr;ii(&R);ii(&gl);ii(&gr);gpp_range(&R,a,b);int ismax=strcmp(sg,"max")==0;int good=ismax?mpfr_sgn(R.hi)<0:mpfr_sgn(R.lo)>0;if(!good){fprintf(stderr,"critical curvature fail %s\n",name);return 5;}mpfr_t margin;mpfr_init2(margin,PREC);if(ismax)mpfr_neg(margin,R.hi,MPFR_RNDD);else mpfr_set(margin,R.lo,MPFR_RNDD);if(mpfr_cmp(margin,min_gpp_margin)<0)mpfr_set(min_gpp_margin,margin,MPFR_RNDD);gp_point2(&gl,a);gp_point2(&gr,b);if(ismax){if(!(mpfr_sgn(gl.lo)>0&&mpfr_sgn(gr.hi)<0)){fprintf(stderr,"max gp fail %s\n",name);return 5;}if(!cs||!rs){fprintf(stderr,"critical center missing\n");return 2;}mpq_t c,rad;mpq_init(c);mpq_init(rad);parse_q(c,cs);parse_q(rad,rs);I gc,gpc;ii(&gc);ii(&gpc);g_point2(&gc,c);gp_point2(&gpc,c);mpfr_t A,m,den,corr,ub,t;mpfr_init2(A,PREC);mpfr_init2(m,PREC);mpfr_init2(den,PREC);mpfr_init2(corr,PREC);mpfr_init2(ub,PREC);mpfr_init2(t,PREC);mpfr_abs(A,gpc.lo,MPFR_RNDU);mpfr_abs(t,gpc.hi,MPFR_RNDU);if(mpfr_cmp(t,A)>0)mpfr_set(A,t,MPFR_RNDU);mpfr_neg(m,R.hi,MPFR_RNDD);mpfr_mul(den,m,m,MPFR_RNDD); /* overwritten below */ mpfr_mul_ui(den,m,2,MPFR_RNDD);mpfr_mul(t,A,A,MPFR_RNDU);mpfr_div(corr,t,den,MPFR_RNDU);mpfr_add(ub,gc.hi,corr,MPFR_RNDU);if(mpfr_sgn(ub)>=0){char*u=mstr(ub);fprintf(stderr,"max g fail %s ub=%s\n",name,u);mpfr_free_str(u);return 5;}if(mpfr_cmp(ub,global_ub)>0)mpfr_set(global_ub,ub,MPFR_RNDU);mpfr_clear(A);mpfr_clear(m);mpfr_clear(den);mpfr_clear(corr);mpfr_clear(ub);mpfr_clear(t);ci(&gc);ci(&gpc);mpq_clear(c);mpq_clear(rad);}else{if(!(mpfr_sgn(gl.hi)<0&&mpfr_sgn(gr.lo)>0)){fprintf(stderr,"min gp fail %s\n",name);return 5;}}mpfr_clear(margin);ci(&R);ci(&gl);ci(&gr);
  } else if(strcmp(kind,"INFL")==0){infl++;if(!cs||!rs){fprintf(stderr,"inflection center missing\n");return 2;}mpq_t c,rad;mpq_init(c);mpq_init(rad);parse_q(c,cs);parse_q(rad,rs);I gpc;ii(&gpc);gp_point2(&gpc,c);mpfr_t M,tmp,delta,rlo,rhi,margin;mpfr_init2(M,PREC);mpfr_init2(tmp,PREC);mpfr_init2(delta,PREC);mpfr_init2(rlo,PREC);mpfr_init2(rhi,PREC);mpfr_init2(margin,PREC);mpfr_set_ui(M,0,MPFR_RNDU);for(int j=0;j<12;j++){mpq_t u,v,w,sj,sj1;mpq_init(u);mpq_init(v);mpq_init(w);mpq_init(sj);mpq_init(sj1);mpq_sub(w,b,a);mpq_set_si(sj,j,12);mpq_set_si(sj1,j+1,12);mpq_mul(u,w,sj);mpq_add(u,u,a);mpq_mul(v,w,sj1);mpq_add(v,v,a);I G;ii(&G);gpp_range(&G,u,v);mpfr_abs(tmp,G.lo,MPFR_RNDU);if(mpfr_cmp(tmp,M)>0)mpfr_set(M,tmp,MPFR_RNDU);mpfr_abs(tmp,G.hi,MPFR_RNDU);if(mpfr_cmp(tmp,M)>0)mpfr_set(M,tmp,MPFR_RNDU);ci(&G);mpq_clear(u);mpq_clear(v);mpq_clear(w);mpq_clear(sj);mpq_clear(sj1);}mpfr_t radhi;mpfr_init2(radhi,PREC);mpfr_set_q(radhi,rad,MPFR_RNDU);mpfr_mul(delta,M,radhi,MPFR_RNDU);mpfr_sub(rlo,gpc.lo,delta,MPFR_RNDD);mpfr_add(rhi,gpc.hi,delta,MPFR_RNDU);int pos=strcmp(sg,"positive")==0;int good=pos?mpfr_sgn(rlo)>0:mpfr_sgn(rhi)<0;if(!good){char*l=mstr(rlo),*h=mstr(rhi);fprintf(stderr,"inflection gp fail %s [%s,%s]\n",name,l,h);mpfr_free_str(l);mpfr_free_str(h);return 5;}if(pos)mpfr_set(margin,rlo,MPFR_RNDD);else mpfr_neg(margin,rhi,MPFR_RNDD);if(mpfr_cmp(margin,min_gp_margin)<0)mpfr_set(min_gp_margin,margin,MPFR_RNDD);mpfr_clear(M);mpfr_clear(tmp);mpfr_clear(delta);mpfr_clear(rlo);mpfr_clear(rhi);mpfr_clear(margin);mpfr_clear(radhi);ci(&gpc);mpq_clear(c);mpq_clear(rad);
  } else if(strcmp(kind,"TAIL")==0){tail++;I R;ii(&R);gp_range2(&R,a,b);int pos=strcmp(sg,"positive")==0;int good=pos?mpfr_sgn(R.lo)>0:mpfr_sgn(R.hi)<0;if(!good){char*l=mstr(R.lo),*h=mstr(R.hi);fprintf(stderr,"tail gp fail %s [%s,%s]\n",name,l,h);mpfr_free_str(l);mpfr_free_str(h);return 5;}mpfr_t margin;mpfr_init2(margin,PREC);if(pos)mpfr_set(margin,R.lo,MPFR_RNDD);else mpfr_neg(margin,R.hi,MPFR_RNDD);if(mpfr_cmp(margin,min_gp_margin)<0)mpfr_set(min_gp_margin,margin,MPFR_RNDD);mpfr_clear(margin);ci(&R);
  } else if(strcmp(kind,"END")==0){endc++;mpfr_t ub;mpfr_init2(ub,PREC);g_box_upper(ub,a,b);if(mpfr_sgn(ub)>=0){fprintf(stderr,"endpoint g fail %s\n",name);return 5;}if(mpfr_cmp(ub,global_ub)>0)mpfr_set(global_ub,ub,MPFR_RNDU);mpfr_clear(ub);
  } else {fprintf(stderr,"unknown kind\n");return 2;}
  fprintf(log,"PASS %s %s\n",kind,name);mpq_clear(a);mpq_clear(b);
 }
 fclose(f);fclose(log);char*gu=mstr(global_ub),*gm=mstr(min_gp_margin),*cm=mstr(min_gpp_margin);FILE*o=fopen(argv[5],"w");fprintf(o,"{\n  \"engine\": \"MPFR %s direct C, 256-bit precision\",\n  \"rows\": %d,\n  \"critical_boxes\": %d,\n  \"inflection_boxes\": %d,\n  \"endpoint_boxes\": %d,\n  \"tails\": %d,\n  \"global_candidate_upper_bound\": \"%s\",\n  \"minimum_gp_sign_margin\": \"%s\",\n  \"minimum_critical_curvature_margin\": \"%s\",\n  \"passed\": true\n}\n",mpfr_get_version(),rows,crit,infl,endc,tail,gu,gm,cm);fclose(o);printf("passed global rows=%d global_ub=%s\n",rows,gu);mpfr_free_str(gu);mpfr_free_str(gm);mpfr_free_str(cm);
 for(int i=0;i<6;i++){ci(&HI[i]);mpq_clear(HQ[i]);}for(int i=0;i<5;i++){ci(&HPI[i]);mpq_clear(HPQ[i]);}for(int i=0;i<4;i++){ci(&HPPI[i]);mpq_clear(HPPQ[i]);}ci(&PI);mpfr_clear(global_ub);mpfr_clear(min_gp_margin);mpfr_clear(min_gpp_margin);return 0;
}
