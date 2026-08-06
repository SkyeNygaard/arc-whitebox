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
int mpfr_set_si(mpfr_ptr, long, mpfr_rnd_t);
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
int mpfr_zero_p(mpfr_srcptr);
void mpfr_set_inf(mpfr_ptr, int);
int __gmpfr_out_str(FILE*, int, size_t, mpfr_srcptr, mpfr_rnd_t);
#define mpfr_out_str __gmpfr_out_str
int mpfr_asprintf(char**, const char*, ...);
void mpfr_free_str(char*);
const char *mpfr_get_version(void);

#define PREC 512
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


#define M 30
static void init_arr(I *a,int n){for(int i=0;i<n;i++){ii(&a[i]);set_ui_i(&a[i],0);}}
static void clear_arr(I *a,int n){for(int i=0;i<n;i++)ci(&a[i]);}
static int is_zero(const I*x){return mpfr_zero_p(x->lo)&&mpfr_zero_p(x->hi);}
static void poly_mul_arr(I*out,const I*a,const I*b){for(int i=0;i<=M;i++)set_ui_i(&out[i],0);for(int i=0;i<=M;i++){if(is_zero(&a[i]))continue;for(int j=0;j+i<=M;j++){if(is_zero(&b[j]))continue;I t,s;ii(&t);ii(&s);mul_i(&t,&a[i],&b[j]);add_i(&s,&out[i+j],&t);copy_i(&out[i+j],&s);ci(&t);ci(&s);}}}
static void invsqrt_series(I *y,const I*z0,int maxm){I one,two,a0,a1,a2,tmp;ii(&one);ii(&two);ii(&a0);ii(&a1);ii(&a2);ii(&tmp);set_ui_i(&one,1);set_ui_i(&two,2);mul_i(&tmp,z0,z0);sub_i(&a0,&one,&tmp);mul_i(&tmp,&two,z0);neg_i(&a1,&tmp);mpfr_set_si(a2.lo,-1,MPFR_RNDD);mpfr_set_si(a2.hi,-1,MPFR_RNDU);I sq;ii(&sq);sqrt_i(&sq,&a0);div_i(&y[0],&one,&sq);ci(&sq);for(int n=1;n<=maxm;n++){I sum;ii(&sum);set_ui_i(&sum,0);int top=n<2?n:2;for(int k=1;k<=top;k++){mpq_t fq;mpq_init(fq);mpq_set_si(fq,k-2*n,2);I fac,prod1,prod2,ns;ii(&fac);ii(&prod1);ii(&prod2);ii(&ns);set_q_i(&fac,fq);const I *ak=(k==1?&a1:&a2);mul_i(&prod1,ak,&y[n-k]);mul_i(&prod2,&fac,&prod1);add_i(&ns,&sum,&prod2);copy_i(&sum,&ns);ci(&fac);ci(&prod1);ci(&prod2);ci(&ns);mpq_clear(fq);}I ni,den;ii(&ni);ii(&den);set_ui_i(&ni,(unsigned long)n);mul_i(&den,&ni,&a0);div_i(&y[n],&sum,&den);ci(&sum);ci(&ni);ci(&den);}ci(&one);ci(&two);ci(&a0);ci(&a1);ci(&a2);ci(&tmp);}
static void kappa_taylor(I*b,const I*z0){I kval,kp;ii(&kval);ii(&kp);kappa_interval(&kval,&kp,z0);copy_i(&b[0],&kval);copy_i(&b[1],&kp);I q[M-1];init_arr(q,M-1);invsqrt_series(q,z0,M-2);for(int j=2;j<=M;j++){I ji,jm1,prod,den;ii(&ji);ii(&jm1);ii(&prod);ii(&den);set_ui_i(&ji,(unsigned long)j);set_ui_i(&jm1,(unsigned long)(j-1));mul_i(&prod,&ji,&jm1);mul_i(&den,&PI,&prod);div_i(&b[j],&q[j-2],&den);ci(&ji);ci(&jm1);ci(&prod);ci(&den);}clear_arr(q,M-1);ci(&kval);ci(&kp);}
static void compose(I*out,const I*g){I b[M+1],u[M+1],power[M+1],next[M+1];init_arr(b,M+1);init_arr(u,M+1);init_arr(power,M+1);init_arr(next,M+1);kappa_taylor(b,&g[0]);for(int i=0;i<=M;i++)copy_i(&u[i],&g[i]);set_ui_i(&u[0],0);set_ui_i(&power[0],1);for(int j=0;j<=M;j++){if(!is_zero(&b[j]))for(int k=0;k<=M;k++){if(is_zero(&power[k]))continue;I t,s;ii(&t);ii(&s);mul_i(&t,&b[j],&power[k]);add_i(&s,&out[k],&t);copy_i(&out[k],&s);ci(&t);ci(&s);}if(j!=M){poly_mul_arr(next,power,u);for(int k=0;k<=M;k++)copy_i(&power[k],&next[k]);}}clear_arr(b,M+1);clear_arr(u,M+1);clear_arr(power,M+1);clear_arr(next,M+1);}
static void sphere_moment(mpq_t q,int k){mpq_set_ui(q,1,1);for(int j=0;j<k;j++){mpq_t f;mpq_init(f);mpq_set_ui(f,2*j+1,256+2*j);mpq_mul(q,q,f);mpq_clear(f);}}
static char* mstr2(const mpfr_t x){char*s=NULL;mpfr_asprintf(&s,"%.120Re",x);return s;}
int main(int argc,char**argv){if(argc!=2){fprintf(stderr,"usage: %s output.json\n",argv[0]);return 2;}mpfr_set_default_prec(512);/* existing I helpers use PREC=256 init; override is ineffective for those. */ii(&PI);mpfr_const_pi(PI.lo,MPFR_RNDD);mpfr_const_pi(PI.hi,MPFR_RNDU);I g[M+1],ng[M+1];init_arr(g,M+1);init_arr(ng,M+1);set_ui_i(&g[1],1);for(int layer=0;layer<32;layer++){for(int i=0;i<=M;i++)set_ui_i(&ng[i],0);compose(ng,g);for(int i=0;i<=M;i++){if(mpfr_sgn(ng[i].hi)<0){fprintf(stderr,"negative coeff upper layer %d degree %d\n",layer+1,i);return 5;}copy_i(&g[i],&ng[i]);}}
 I partial;ii(&partial);set_ui_i(&partial,0);for(int k=0;k<=M/2;k++){mpq_t mq;mpq_init(mq);sphere_moment(mq,k);I mi,term,sum;ii(&mi);ii(&term);ii(&sum);set_q_i(&mi,mq);mul_i(&term,&g[2*k],&mi);add_i(&sum,&partial,&term);copy_i(&partial,&sum);ci(&mi);ci(&term);ci(&sum);mpq_clear(mq);}mpq_t tq;mpq_init(tq);sphere_moment(tq,M/2+1);I tail;ii(&tail);set_q_i(&tail,tq);mpfr_t upper,width;mpfr_init2(upper,PREC);mpfr_init2(width,PREC);mpfr_add(upper,partial.hi,tail.hi,MPFR_RNDU);mpfr_sub(width,upper,partial.lo,MPFR_RNDU);char*l=mstr2(partial.lo),*u=mstr2(upper),*w=mstr2(width),*tu=mstr2(tail.hi);FILE*o=fopen(argv[1],"w");fprintf(o,"{\n  \"engine\": \"MPFR %s direct C interval Taylor jets\",\n  \"precision_bits\": %d,\n  \"dimension\": 256,\n  \"depth\": 32,\n  \"maximum_computed_degree\": 30,\n  \"A0_lower\": \"%s\",\n  \"A0_upper\": \"%s\",\n  \"A0_width\": \"%s\",\n  \"tail_moment_upper\": \"%s\",\n  \"passed\": true\n}\n",mpfr_get_version(),PREC,l,u,w,tu);fclose(o);printf("A0=[%s,%s] width=%s\n",l,u,w);mpfr_free_str(l);mpfr_free_str(u);mpfr_free_str(w);mpfr_free_str(tu);mpq_clear(tq);ci(&tail);mpfr_clear(upper);mpfr_clear(width);ci(&partial);clear_arr(g,M+1);clear_arr(ng,M+1);ci(&PI);return 0;}
